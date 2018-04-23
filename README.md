# roomnet
This is a tensorflow implementation of room layout paper: [RoomNet: End-to-End Room Layout Estimation](https://arxiv.org/pdf/1703.06241.pdf).

**Note**: This is a simply out-of-interest experiemnt and I cannot guarantee to get the same effect of the origin paper.

## Network
![Roomnet network Architecture](https://github.com/GitBoSun/roomnet/blob/master/images/net.png)
Here I implement two nets: vanilla encoder-decoder version and 3-iter RCNN refined version. As the author noted, the latter achieve better results.

## Data
I use [LSUN dataset](http://lsun.cs.princeton.edu/2017/) and please download and prepare the RGB images and get a explorationo of the .mat file it includs because they contain layout type, key points and other information.
Here I simply resize the image to (320, 320) with cubic interpolation and do the flip horizontally. (**Note**: When you flip the image, the order of layout key points should also be fliped.) You can see the preparation of data in [prepare_data.py]()

## Pre-requests:
You need to install tensorflow>=1.2, opencv, numpy, scipy and other basic dependencies.

## How to use:
Training: 
```
python main.py --train 0 or 1 --net vanilla ot rcnn --out_path path-to-output 
```
Testing:
```
python main.py --test 0 --net vanilla ot rcnn --out_path path-to-output 
```
(P.S. I may upload the pre-trained model later because currently I don't find a place to put it.)

## Some Results:
