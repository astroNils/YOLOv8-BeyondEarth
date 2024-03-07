import cv2
import numpy as np
import pandas as pd
import torch

from YOLOv8BeyondEarth.polygon import (check_mask_validity, binary_mask_to_polygon, is_within_slice, shift_polygon,
                                       add_geometries, bboxes_to_shp, outlines_to_shp)
from lsnms import nms, wbc
from sahi.slicing import slice_image
from tqdm import tqdm
from pathlib import Path

from rastertools_BOULDERING import raster, convert as raster_convert

def YOLOv8(detection_model, image, has_mask, shift_amount, slice_height, slice_width, verbose):
    """
    YOLOv8 expects numpy arrays to have BGR (height, width, 3).

    1. Let's say you want to detect very very small objects, the slice height and width should be
    pretty small, and detection_model.image_size should be increased:
    - slice_height, slice_width = 256
    - detection_model.image_size = 1024.

    2. If you are in the opposite situation, where you realized that most of the large boulders are missed out. You can
    increase the slice height and width and decrease the detection_model.image_size.
    - slice_height, slice_width = 1024
    - detection_model.image_size = 256 or 512.

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

                # resizing from model.image_size to slice_width and slice_height
                bool_mask = cv2.resize(bool_mask, (slice_width, slice_height))  # [1,0]
                bool_mask[bool_mask >= 0.5] = 1
                bool_mask[bool_mask < 0.5] = 0

                is_binary_mask_valid = check_mask_validity(bool_mask)
                if is_binary_mask_valid:
                    polygon = np.array(binary_mask_to_polygon(bool_mask, tolerance=0)).squeeze()

                    # are the coordinates of the polygon touching the edge of the slice?
                    is_polygon_within_slice = is_within_slice(polygon, slice_width, slice_height)

                    if not is_within_slice(polygon, slice_width, slice_height):
                        score = 0.10  # if at edge set score to a low value

                    # conversion to absolute coordinates,
                    shifted_polygon = shift_polygon(polygon, shift_x, shift_y)

                    scores.append(score)
                    polygons.append(shifted_polygon)  # conversion of polygon to absolute coordinates
                    category_ids.append(category_id)
                    category_names.append(category_name)
                    is_polygon_within_slice_list.append(is_polygon_within_slice)

        dict = {'score': scores, 'polygon': polygons,
                'category_id': category_ids, 'category_name': category_names,
                'is_within_slice': is_polygon_within_slice_list}

        df = pd.DataFrame(dict)
    return df

def get_sliced_prediction(in_raster,
                          detection_model=None,
                          has_mask=True,
                          output_dir=None,
                          interim_file_name=None,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
                          interim_dir=None,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
                          slice_height: int = None,
                          slice_width: int = None,
                          inference_size: int = None,
                          overlap_height_ratio: float = 0.2,
                          overlap_width_ratio: float = 0.2,
                          postprocess: bool = True,
                          postprocess_match_threshold: float = 0.5,
                          postprocess_class_agnostic: bool = False):
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        in_raster: str or Path()
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
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

    # inference size
    detection_model.image_size = inference_size

    slice_image_result = slice_image(
        image=out_png.as_posix(),  # need to be path to
        output_file_name=interim_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        out_ext=".png",  # FORMAT OF (OPTIONALLY) SAVED SLICES
    )

    num_slices = len(slice_image_result)
    shift_amounts = slice_image_result.starting_pixels

    # create prediction input
    tqdm.write(f"Performing prediction on {num_slices} number of slices.")

    frames = []
    # perform sliced prediction
    for i, image in tqdm(enumerate(slice_image_result.images), total=num_slices):
        df = YOLOv8(detection_model, image, has_mask, shift_amounts[i], slice_height, slice_width, verbose)
        if df.shape[0] > 0:
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    gdf = add_geometries(in_raster, df_all)

    # save shapefile before post-processing
    out_bbox_shp = output_dir / (in_raster.stem + "-predictions-ss-" + str(slice_height) + "-is-" +
                                 str(inference_size) + "-ov-" + str(int(overlap_height_ratio*100)).zfill(3) + "-bbox.shp")
    out_mask_shp = output_dir / (in_raster.stem + "-predictions-ss-" + str(slice_height) + "-is-" +
                                 str(inference_size) + "-ov-" + str(int(overlap_height_ratio*100)).zfill(3) + "-mask.shp")
    bboxes_to_shp(gdf, out_bbox_shp)
    outlines_to_shp(gdf, out_mask_shp)


    if postprocess:
        # processing (NMS)
        if postprocess_class_agnostic:
            keep = nms(np.stack(gdf.bbox.values), gdf.score.values,
                       iou_threshold=postprocess_match_threshold)  # no difference between categories
        else:
            keep = nms(np.stack(gdf.bbox.values), gdf.score.values, iou_threshold=postprocess_match_threshold,
                       class_ids=gdf.category_id.values)

        # saving post-processed shapefiles
        gdf_nms = gdf.loc[keep]
        bboxes_to_shp(gdf_nms, out_bbox_shp.with_name(out_bbox_shp.stem + "-nms.shp"))
        outlines_to_shp(gdf_nms, out_mask_shp.with_name(out_mask_shp.stem + "-nms.shp"))
    else:
        None