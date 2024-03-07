import cv2
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("C:/Users/nilscp/GIT/YOLOv8BeyondEarth/src/")

from YOLOv8BeyondEarth.polygon import (check_mask_validity, binary_mask_to_polygon, is_within_slice, shift_polygon)

def YOLOv8(detection_model, image, has_mask, shift_amount, slice_height, slice_width, verbose):
    """
    YOLOv8 expects numpy arrays to have BGR (height, width, 3).

    Let's way you want to detect very very small objects, the slice height and width should be
    pretty small, and detection_model.image_size should be increased slice_height, slice_width = 256, and
    detection_model.image_size = 512 or 1024.

    The bboxes are calculated from the polygons after the predictions are computed.
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