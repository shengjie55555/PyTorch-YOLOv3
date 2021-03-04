# PyTorch-YOLOv3

## Introduction

This repo is an implementation of YOLOv3 with PyTorch 1.7. Currently only detection is finished and the training part will be added in the future.

## How it works?

I wrote a detailed tutorial. If you are interested, you can click [here](https://www.zhihu.com/column/c_1349455968032374785) check it.

<!--The tutorial is written in Chinese!-->

## How to use it?

### Environment

- CUDA 10.1
- CUDNN 7.6.5
- ubuntu 20.04
- python 3.8.5
- PyTorch 1.7

### Prepared work

1. Git clone this repository

```shell
git clone https://github.com/shengjie55555/PyTorch-YOLOv3.git
```

2. Download weights file

```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

After downloading, put it in the main directory.

3. Test

- Run detector.py to detect an image.
- Run video.py to detect a video or open your camera to detect object in real time.

## To do

1. Training.
2. Add some tricks to improve the model.

