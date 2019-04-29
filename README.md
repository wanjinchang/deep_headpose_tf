# Deep_headpose-tensorflow

This is an implementation of [Fine-Grained Head Pose Estimation Without Keypoints
](https://arxiv.org/abs/1710.00925).  

### Prerequisites

1. You need a CUDA-compatible GPU to train the model.
2. You should first download [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) for face detection.

### Dependencies

* TensorFlow 1.4.1
* TF-Slim
* Python3.6
* Ubuntu 16.04
* Cuda 8.0

### Contents

1. [Setup_data](#setup_data)
2. [Training](#training)
3. [Demo](#demo)
4. [Models](#models)

## Setup_data

Usr the script ``python utils/data_preprocess.py`` to generate your own annotation file and face images from 300W-LP dataset.
    One row for one image;  
    Row format: `image_file_path pitch raw roll`;  
    Here is an example:
```
LFPW_image_test_0236_16_1.jpg,-83.1405049927369,-12.156124797978574,-2.719337151759615
LFPW_image_train_0095_5_2.jpg,-31.33569213834624,-8.705391993430364,19.86217644601874
...
```
Or you can use my annotation files `300W_LP_headpose_anno.txt` under the folder ``data/`` directly.  

## Training

-  Download pre-trained models and weights of backbones.The current code supports VGG16/ResNet_V1/MobileNet_Series models. 
-  Pre-trained models are provided by slim, you can get the pre-trained models from [Google Driver](https://drive.google.com/open?id=1iqOZNA9nwvITvwTDvK2gZUHAI1fo_XHI) or [BaiduYun Driver](https://pan.baidu.com/s/1m7uv9Sqs6hEb3VcMy3gFzg). Uzip and place them in the folder ``data/imagenet_weights``. For example, for VGG16 model, you also can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For ResNet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

-  Train
  Run `python trainval_net.py` to train the model. 
  ```
  python trainval_net.py \
      --cfg=experiment/cfg/mobile.yml \
      --weight=data/imagenet_weights/mobile.ckpt \
      --gpu_id=0 \
      --data_dir=/home/oeasy/Downloads/dataset/head_pose/300W_LP_headpose \
      --annotation_path=/home/oeasy/PycharmProjects/deep-head-pose-tf/300W_LP_headpose_anno.txt \
      --net=mobile
  ```
  Please see details in the script `trainval_net.py`.
By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

-  For ckpt demo
Download trained models from [Models](#models), then uzip to the folder ``output/``, modify your path of trained model
   run ``python tools/demo.py`` directly.

-  For frozen graph inference
Download the pb models(contained in [Models](#models)) or frozen your model by yourself using script ``tools/convert_ckpt_to_pb.py``, modify your path of trained model, then run ``python tools/inference.py``.

## Models
  Under updating...
