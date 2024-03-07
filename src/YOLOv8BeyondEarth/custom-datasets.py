import numpy as np
import skimage
import pycocotools.mask as mask_util

from tqdm import tqdm
from YOLOv8BeyondEarth.polygon import binary_mask_to_polygon

# for polygon to detectron2 (see MLtools github repo).
# for masks2yolo modify the code below.
def detectron2yolo(detectron2_json, min_area_threshold, max_area_threshold, pre_processed_folder):
    """
    the detectron2_json needs to have a 'dataset' column.
    max_area_threshold is to remove too large objects for the tile.

    This function generates text file following the Ultralytics YOLO format
    (https://docs.ultralytics.com/datasets/segment/#supported-dataset-formats)
    """
    for i, row in tqdm(detectron2_json.iterrows(), total=detectron2_json.shape[0]):
        masks = []
        for r in row.annotations:
            rle = r["segmentation"]
            # to avoid holes within mask
            masks.append(skimage.morphology.remove_small_holes(mask_util.decode(rle)))

        contours = []
        for m in masks:
            npixels = len(m[m == 1])
            # min, max area
            if np.logical_and(npixels > min_area_threshold, npixels < max_area_threshold):
                contours.append(binary_mask_to_polygon(m))

        txt_stem = row.file_name.replace("png", "txt")
        txt_filename = pre_processed_folder / row.dataset / "labels" / txt_stem
        with open(txt_filename.as_posix(), "a") as f:
            for contour in contours:
                c = np.array(contour).squeeze()
                cc = np.stack([c[:, 0], c[:, 1]], axis=-1)
                arr = cc.flatten() / row.height
                arr = list(arr.round(decimals=3))
                arr.insert(0, 0)
                str_arr = [str(a) for a in arr]
                line = " ".join(str_arr) + "\n"
                f.writelines(line)

