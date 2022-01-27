# snst-img
### Selective Neural Style Transfer in images

#### Usage
`$ python snst_img.py -i [IMAGE] -s [STYLE IMAGE] -m [POINTREND MODEL] -o [OUTPUT FILE]` \
For more info, use `-h` flag.

#### Requirements
Note: Install Tensorflow and PyTorch libraries built for your respective hardware (ie. CUDA GPUs).
* `opencv-python>=4.5.5.62`
* `numpy>=1.21.5`
* `Pillow>=9.0.0`
* `matplotlib>=3.5.1`
* `tensorflow>=2.7.0`
* `tensorflow-hub>=0.12.0`
* `torch>=1.10.1`
* `pixellib>=0.7.1`
* PointRend [model](https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl) for image segmentation
