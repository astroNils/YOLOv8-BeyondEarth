# YOLOv8-BeyondEarth
Tools and scripts to create YOLOv8 custom datasets, train the model and post-process the obtained results. Strong focus on the use of satellite imagery, and application on soild planetary bodies in our Solar system.

## To do

- [ ] There is currently an issue with the non-maximum suppression from the lsnms github repository (https://github.com/remydubois/lsnms/issues/29). It does not filter the overlapping bounding boxes in the correct way. I have posted an issue on the repo. I need to follow-up if there is an answer.
- [ ] Adapt `get_sliced_prediction` for use with Mask R-CNN. 
- [ ] The `postprocess_class_agnostic` flag is currently not working (so basically the NMS step is working only for cases where the predictions are made for one object class).
- [ ] Use of batches for prediction could speed up the prediction. 

## Installation

Create a new environment if wanted. Then you can install the rastertools by writing the following in your terminal. Before installing YOLOv8-BeyondEarth, you need to install `rastertools` and `shptools`.

#### rastertools

```bash
git clone https://github.com/astroNils/rastertools.git
cd rastertools
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING
pip install -r requirements.txt
```

#### shptools

```bash
git clone https://github.com/astroNils/shptools.git
cd shptools
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING
pip install -r requirements.txt
```

#### YOLOv8-BeyondEarth

````bash
git clone https://github.com/astroNils/YOLOv8-BeyondEarth
cd YOLOv8-BeyondEarth
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps YOLOv8BeyondEarth
pip install -r requirements.txt
````

#### sahi

To get the last version of sahi.

```bash
git clone https://github.com/obss/sahi.git
cd sahi
pip install -e . 
```

You should now have access to this module in Python.

```bash
python
```

```python
from YOLOv8BeyondEarth.predict import get_sliced_prediction
```

## Getting Started

A jupyter notebook is provided as a tutorial ([GET STARTED HERE](./resources/nb/GETTING_STARTED.ipynb)).

### Few links (YOLO, YOLOv8, anchor-free object detection)

https://github.com/ultralytics/ultralytics

https://medium.com/cord-tech/yolov8-for-object-detection-explained-practical-example-23920f77f66a

https://keylabs.ai/blog/comparing-yolov8-and-yolov7-whats-new/

https://learnopencv.com/fcos-anchor-free-object-detection-explained/

https://learnopencv.com/centernet-anchor-free-object-detection-explained/#What-is-Anchor-Free-Object-Detection

https://www.datacamp.com/blog/yolo-object-detection-explained
