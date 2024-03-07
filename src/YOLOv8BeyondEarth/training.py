from ultralytics import YOLO
import os

os.chdir("D:/BOULDERING/data/yolov8/training/")

#Instance
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8n-seg.pt')  # Transfer the weights from a pretrained model (recommended for training, here from ImageNet)

results = model.train(data="D:/BOULDERING/data/yolov8/datasets/boulder2024/boulder2024.yaml",
                      project="YOLOv8-training",
                      name="YOLOv8-200-epochs-19-02-2024-1024x1024",
                      save_dir="D:/BOULDERING/data/yolov8/training/",
                      epochs=200, # (int) number of epochs to train for
                      patience=0, #I am setting patience=0 to disable early stopping.
                      batch=4, # (int) number of images per batch (-1 for AutoBatch)
                      imgsz=1024, # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
                      save=True,
                      save_period=20, # (int) Save checkpoint every x epochs (disabled if < 1)
                      device=0,
                      workers=0,
                      exist_ok=False, # (bool) whether to overwrite existing experiment
                      optimizer="Adam", # wonder if SGD would go faster
                      lr0=0.001,
                      val=True,
                      plots=True,
                      verbose=True,
                      resume=False,
                      cache=False, # (bool) True/ram, disk or False. Use cache for data loading
                      seed=0, # (int) random seed for reproducibility
                      close_mosaic=10, # (int) disable mosaic augmentation for final epochs (0 to disable)
                      overlap_mask=True, # (bool) masks should overlap during training (segment train only)
                      iou=0.7, # (float) intersection over union (IoU) threshold for NMS
                      mask_ratio=1, # I don't want any downsampling, it was actually giving better results with 4? not sure why
                      max_det=2000, # (int) maximum number of detections per image
                      translate=0.1, # image translation (+/- fraction)
                      scale=0.5, # image scale (+/- gain)
                      shear=0.0, # image shear (+/- deg)
                      perspective=0.0, # (float) image perspective (+/- fraction), range 0-0.001
                      flipud=0.0, # image flip up-down (probability)
                      fliplr=0.5, # image flip left-right (probability)
                      mosaic=1.0, # image mosaic (probability)
                      mixup=0.0, # image mixup (probability)
                      copy_paste=0.0, # segment copy-paste (probability)
                      erasing=0.4) # let's upscale from 500 to 1024