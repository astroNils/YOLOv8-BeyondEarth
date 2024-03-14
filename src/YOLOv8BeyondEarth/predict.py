import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from YOLOv8BeyondEarth.polygon import (binary_mask_to_polygon, is_within_slice, shift_polygon,
                                       add_geometries, bboxes_to_shp, outlines_to_shp)
from lsnms import nms, wbc

from sahi.slicing import slice_image
from tqdm import tqdm
from pathlib import Path

from rastertools_BOULDERING import raster, convert as raster_convert, metadata as raster_metadata
from shptools_BOULDERING import shp

#from torchvision.ops import (nms as nms_torch, batched_nms as batched_nms_torch)

def YOLOv8(detection_model, image, has_mask, shift_amount, slice_size, min_area_threshold, downscale_pred):
    """
    YOLOv8 expects numpy arrays to have BGR (height, width, 3).

    1. Let's say you want to detect very very small objects, the slice height and width should be
    pretty small, and detection_model.image_size should be increased:
    - slice_height, slice_width = 256
    - detection_model.image_size = 1024.

    2. If you are in the opposite situation, where you realized that most of the large boulders are missed out. You can
    increase the slice height and width.
    - slice_height, slice_width = 1024
    - detection_model.image_size = 512, 1024.

    You can get the best of both worlds by combining predictions (1) and (2) with NMS. Obviously the larger the
    slices height and width, and the larger the detection_model.image_size, the more time it takes to run this
    script.

    If the predictions is starting to be very large compare to the size of the slice, WBF can be advantageous as it
    will merge the overlapping boulders. However, WBF and NMS works better in less dense area (or at least is less
    sensitive to the iou_threshold selected).

    Test Time Augmentation could be included too, but it takes lot of time to run it.. so not sure about that too.
    WBF for instance seg: "https://www.kaggle.com/code/mistag/sartorius-tta-with-weighted-segments-fusion"

    Note that the bboxes (in absolute coordinates) are calculated from the bounds of the polygons after the
    predictions are computed.
    """

    shift_x = shift_amount[0]
    shift_y = shift_amount[1]

    prediction_results = detection_model.model(image, imgsz=detection_model.image_size, verbose=verbose,
                                               device=detection_model.device)

    # if no predictions
    if prediction_results[0].boxes.data.size()[0] == 0:
        df = pd.DataFrame(columns=['score', 'polygon', 'category_id', 'category_name', 'is_within_slice'])
    else:
        if has_mask:
            predictions = [
                (result.boxes.data[result.boxes.data[:, 4] >= detection_model.confidence_threshold],
                 result.masks.data[result.boxes.data[:, 4] >= detection_model.confidence_threshold],)
                for result in prediction_results]

        else:
            predictions = []
            for result in prediction_results:
                result_boxes = result.boxes.data[result.boxes.data[:, 4] >= detection_model.confidence_threshold]
                result_masks = torch.tensor([[] for _ in range(result_boxes.size()[0])])
                predictions.append((result_boxes, result_masks))

        # for one image
        # bboxes = [], dropping it as I am calculating it later on.
        scores = []
        polygons = []
        category_ids = []
        category_names = []
        is_polygon_within_slice_list = []

        # names are very confusing I should fix that
        for image_ind, image_predictions in enumerate(predictions):
            image_predictions_in_xyxy_format = image_predictions[0]
            image_predictions_masks = image_predictions[1]

            for prediction, bool_mask in zip(
                    image_predictions_in_xyxy_format.cpu().detach().numpy(),
                    image_predictions_masks.cpu().detach().numpy()
            ):

                score = prediction[4]
                category_id = int(prediction[5])
                category_name = detection_model.category_mapping[str(category_id)]

                # more accurate to have this operation before the eventual resizing
                # takes a little bit of extra computational time
                bool_mask[bool_mask >= 0.5] = 1
                bool_mask[bool_mask < 0.5] = 0

                if downscale_pred:
                    if bool_mask.shape[0] == slice_size:
                        None
                    else:
                        bool_mask = cv2.resize(bool_mask, (slice_size, slice_size), interpolation=cv2.INTER_AREA)

                # number of pixels
                area = len(bool_mask[bool_mask == 1])

                if area > min_area_threshold:
                    try:
                        polygon = binary_mask_to_polygon(bool_mask, tolerance=0)
                        if downscale_pred:
                            polygon_slice = polygon
                        else:
                            polygon_slice = np.stack([(polygon[:, 0] / bool_mask.shape[0]) * slice_size,
                                                      (polygon[:, 1] / bool_mask.shape[0]) * slice_size], axis=-1)
                        min_edge_distance = 0.05 * slice_size
                        max_edge_distance = 0.95 * slice_size
                        is_polygon_within_slice = (np.logical_and(polygon_slice[:, 0].min() >= min_edge_distance,
                                                                  polygon_slice[:, 0].max() <= max_edge_distance) and
                                                   np.logical_and(polygon_slice[:, 1].min() >= min_edge_distance,
                                                                  polygon_slice[:, 1].max() <= max_edge_distance))

                        if not is_polygon_within_slice:
                            score = 0.10  # if at edge set score to a low value

                        # conversion to absolute coordinates,
                        shifted_polygon = shift_polygon(polygon_slice, shift_x, shift_y)

                        scores.append(score)
                        polygons.append(shifted_polygon)  # conversion of polygon to absolute coordinates
                        category_ids.append(category_id)
                        category_names.append(category_name)
                        is_polygon_within_slice_list.append(is_polygon_within_slice)
                    except:
                        None

        dict = {'score': scores, 'polygon': polygons,
                'category_id': category_ids, 'category_name': category_names,
                'is_within_slice': is_polygon_within_slice_list}

        df = pd.DataFrame(dict)
    return df

def get_sliced_prediction(in_raster,
                          detection_model=None,
                          confidence_threshold: float = 0.1,
                          has_mask=True,
                          output_dir=None,
                          interim_file_name=None,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
                          interim_dir=None,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
                          slice_size: int = None,
                          inference_size: int = None,
                          overlap_height_ratio: float = 0.2,
                          overlap_width_ratio: float = 0.2,
                          min_area_threshold: int = None,
                          downscale_pred: bool = False,
                          postprocess: bool = True,
                          postprocess_match_threshold: float = 0.5,
                          postprocess_class_agnostic: bool = False):
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    The time to run the script is dependent on the number of predictions over the whole image. This is because we need
    to loop through each prediction and transform the bool_mask to polygon. 

    Args:
        in_raster: str or Path()
            Path to raster tif file.
        detection_model: model.DetectionModel
        confidence_threshold: float
            minimum confidence threshold, values below will be automatically filtered away.
        has_mask: bool
        interim_dir: str or Path()
        slice_height: int
            Height of each slice.  Defaults to ``None``.
        slice_width: int
            Width of each slice.  Defaults to ``None``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        postprocess: bool
            Include postprocessing or not.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.

    Returns:
        A pd.DataFrame.
    """

    # convert in_raster tif file to png file
    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    out_png = in_raster.with_name(in_raster.stem + ".png")
    raster_convert.tiff_to_png(in_raster, out_png)

    # create temporary directory
    tmp_dir = (Path.home() / "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # set model's confidence_threshold and inference size
    detection_model.image_size = inference_size
    detection_model.confidence_threshold = confidence_threshold

    slice_image_result = slice_image(
        image=out_png.as_posix(),  # need to be path to
        output_file_name=interim_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        out_ext=".png",  # FORMAT OF (OPTIONALLY) SAVED SLICES
    )

    num_slices = len(slice_image_result)
    shift_amounts = slice_image_result.starting_pixels
    frames = []
    # perform sliced prediction
    for i, image in tqdm(enumerate(slice_image_result.images), total=num_slices):
        df = YOLOv8(detection_model, image, has_mask, shift_amounts[i], slice_size, min_area_threshold,
                               downscale_pred)
        if df.shape[0] > 0:
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    gdf = add_geometries(in_raster, df_all)

    # keep edge predictions (within 10% of slice size from the true footprint edge)

    # extract footprint
    raster.true_footprint(in_raster, tmp_dir / "true-footprint.shp")
    in_res = raster_metadata.get_resolution(in_raster)[0]
    gdf_true_footprint = gpd.read_file(tmp_dir / "true-footprint.shp")
    gdf_true_footprint = gpd.GeoDataFrame(geometry=[gdf_true_footprint.unary_union.convex_hull],
                                          crs=gdf_true_footprint.crs)
    gdf_true_footprint.to_file(tmp_dir / "true-footprint.shp")
    gpd.GeoDataFrame(geometry=gdf_true_footprint.geometry.boundary.values, crs=gdf_true_footprint.crs).to_file(
        tmp_dir / "true-footprint-as-a-line.shp")
    gdf_line_buffer = shp.buffer(tmp_dir / "true-footprint-as-a-line.shp", slice_size * 0.10 * in_res,
                                 (tmp_dir / "footprint-buffer.shp"))

    gdf_boulders = gdf.copy()
    gdf_boulders["id"] = gdf_boulders.index
    gdf_intersected = gpd.overlay(gdf_boulders, gdf_line_buffer, how="intersection", keep_geom_type=True)

    gdf["is_at_edge"] = False
    gdf.loc[gdf_intersected.id.values, "is_at_edge"] = True

    # keep edge predictions close to the edge of the footprint of the raster, but otherwise remove edge predictions
    gdf = gdf.loc[np.logical_or(gdf.is_at_edge == True, gdf.is_within_slice == True)]

    # remove duplicates
    gdf = gdf.drop_duplicates(subset="geometry", ignore_index=True)
    gdf["id"] = gdf.index

    # save shapefile before post-processing (include if downscaling is done or not...)
    bbox_filename = in_raster.stem + "-predictions-ct-" + str(int(confidence_threshold * 100)).zfill(3) + "-ss-" + str(
        slice_size) + "-is-" + str(inference_size) + "-ov-" + str(int(overlap_height_ratio * 100)).zfill(3) + "-bbox.shp"
    mask_filename = bbox_filename.replace("-bbox.shp", "-mask.shp")

    if downscale_pred:
        bbox_filename = bbox_filename.replace("-bbox.shp", "-downscaled-bbox.shp")
        mask_filename = mask_filename.replace("-mask.shp", "-downscaled-mask.shp")

    out_bbox_shp = output_dir / bbox_filename
    out_mask_shp = output_dir / mask_filename
    bboxes_to_shp(gdf, out_bbox_shp)
    outlines_to_shp(gdf, out_mask_shp)


    if postprocess:
        # Non-maximum suppression (NMS)
        # Note that I have a few issues with nms from LSNMS.. I am currently using wbc from LSNMS which is a
        # workaround to get the correct results. Below, I have commented a few lines showing how to do the NMS
        # step with torchvision or nms from lsnms.

        # regardless of the classes ids (right now is a class agnoistic not supported)
        if postprocess_class_agnostic:
            #keep = nms(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
            #           iou_threshold=postprocess_match_threshold, class_ids=None, rtree_leaf_size=32)

            #keep = nms_torch(boxes=torch.tensor(np.stack(gdf.bbox.values)), scores=torch.tensor(gdf.score.values),
            #                 iou_threshold=postprocess_match_threshold)

            pooled_boxes, pooled_scores, cluster_indices = wbc(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
                                                               iou_threshold=postprocess_match_threshold)
        # or taking into account the classes ids
        else:
            #keep = nms(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
            #           iou_threshold=postprocess_match_threshold, class_ids=gdf.category_id.values, rtree_leaf_size=32)

            #keep = batched_nms_torch(boxes=torch.tensor(np.stack(gdf.bbox.values)), scores=torch.tensor(gdf.score.values),
            #                 idxs=torch.tensor(gdf.category_id.values), iou_threshold=postprocess_match_threshold)

            pooled_boxes, pooled_scores, cluster_indices = wbc(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
                                                               iou_threshold=postprocess_match_threshold)

        keep = np.array([a for a, b in cluster_indices])

        # saving post-processed shapefiles
        gdf_nms = gdf.loc[keep]
        bboxes_to_shp(gdf_nms, out_bbox_shp.with_name(out_bbox_shp.stem + "-nms.shp"))
        outlines_to_shp(gdf_nms, out_mask_shp.with_name(out_mask_shp.stem + "-nms.shp"))
        return gdf, gdf_nms
    else:
        return gdf, None