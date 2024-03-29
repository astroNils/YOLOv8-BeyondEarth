# YOLOv8-BeyondEarth
YOLOv8-BeyondEarth is a repository that contains tools and scripts to create YOLOv8 custom datasets, train the model, and post-process the obtained results. There is a strong focus on the use of satellite imagery and the application of the YOLOv8 model on solid planetary bodies in our Solar system.

## To do

- [ ] Adapt `get_sliced_prediction` for use with Mask R-CNN. 
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

## Bugs

If in your coordinate system, the latitude of origin is different than 0 degrees and this is expressed by the parameter `["Latitude of natural origin"`, there is a bug in the alignment of the prediction shapefiles (i.e., the shapefiles do not align with the raster/image). This bug does not happen for the Lunar Reconnaissance Orbiter NAC images. I wonder if this is due to the previously mentioned coordinate system/projection parameter and some bugs with rasterio/shapely? (need to do more digging there). When this parameter is not defined (i.e., a latitude of origin equal to 0), **then there is no bug**. Note that in NAC images, an equirectangular projection can be centered on a latitude without causing a bug. Looking at the coordinate system, the centering on the latitude seems to be described by another parameter (`["Latitude of 1st standard parallel"`). Therefore,  one solution could be to convert the coordinate system so that the centered latitude is described by the 1st standard parallel (or just keep the center latitude at 0 degrees). You can switch between coordinate systems with the help of `gdalwarp` (https://gdal.org/programs/gdalwarp.html).

```bash
gdalwarp -s_src <old_coordinate_system> -t_srs <wkt_string_where_only_latitude_1st_standard_parallel_is_used> <src_raster> <dst_raster>
```

You can get information about the old coordinate system with the help of `gdal_info`. 

```bash
gdalinfo <src_raster>
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
