#!/usr/bin/env python

# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import time

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
# from nets.darknet53 import Darknet53
from nets.mobilenet_v2.mobilenet_v2 import mobilenetv2
from nets.squeezenet_v1_1 import squeezenet_v1

NETS = {'mobile': ('mobile_hopenet_epoch_45.ckpt',), 'res50': ('res50_hopenet_epoch_230.ckpt',),
        'squeezenetv1': ('squeezenetv1_hopenet_epoch_235.ckpt',)}
DATASETS= {'Pose_300W_LP': ('Pose_300W_LP',)}

def demo(sess, net, img_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    once_time = 0

    im = cv2.imread(img_path)
    im = cv2.resize(im, (227, 227))
    # im = im[np.newaxis, :, :, :]
    t = time.time()
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    print('subtract consume time {}s'.format(time.time() - t))
    im = im_orig[np.newaxis, :, :, :]
    # print('>>>>>>>', im.shape[0], im.shape[1])

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    yaw, pitch, roll, yaw_raw, pitch_raw, roll_raw = net.test_image(sess, im)
    # yaw, pitch = net.test_image(sess, im)
    print(yaw, pitch, roll)
    # print(yaw_raw)
    # print(pitch_raw)
    # print(roll_raw)
    timer.toc()
    once_time = timer.total_time
    print('Detection took {:.3f}s'.format(timer.total_time))

    # cv2_vis(im, CLASSES[1], dets, result_file)
    return yaw, pitch, roll, once_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Headpose demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101 res50 mobile squeezenetv1]',
                        choices=NETS.keys(), default='mobile')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [Pose_300W_LP]',
                        choices=DATASETS.keys(), default='Pose_300W_LP')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('/home/oeasy/PycharmProjects/deep-head-pose-tf/output', demonet, DATASETS[dataset][0], 'default_alpha_0.01',
                              NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.07
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        print('hopenet base_network is vgg16')
        net = vgg16()
    elif demonet == 'res101':
        print('hopenet base_network is resnet101')
        net = resnetv1(num_layers=101)
    elif demonet == 'res50':
        print('hopenet base_network is resnet50')
        net = resnetv1(num_layers=50)
    elif demonet == 'mobile':
        print('hopenet base_network is mobilenet_v1')
        net = mobilenetv1()
    elif demonet == 'mobile_v2':
        print('hopenet base_network is mobilenet_v2')
        net = mobilenetv2()
    elif demonet == 'squeezenetv1':
        print('hopenet base_network is squeezenetv1')
        net = squeezenet_v1()
    else:
        raise NotImplementedError

    net.create_architecture("TEST", interval=3, num_bins=66, tag='default')
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = os.listdir(os.path.join(cfg.DATA_DIR, 'demo'))
    model_predtxt = open('../mobile_Pose_300W_LP_test_AFLW2000_headpose_alpha_0.01.txt', 'w')
    data_path = '/home/oeasy/Downloads/dataset/head_pose/AFLW2000_headpose'
    # model_predtxt = open('../mobile_Pose_300W_LP_test_all_side_faces_alpha_1.txt', 'w')
    # data_path = '/home/oeasy/Downloads/dataset/head_pose/side_face_data/all_side_faces'
    im_names = os.listdir(data_path)
    print('>>>>>', im_names)
    while True:
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            img_path = os.path.join(data_path, im_name)
            print('Demo for {}'.format(img_path))
            yaw, pitch, roll, _ = demo(sess, net, img_path)
            txt = im_name + ',' + str(yaw) + ',' + str(pitch) + ',' + str(roll) + '\n'
            model_predtxt.write(txt)
        model_predtxt.close()
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('Demo for data/demo/{}'.format('demo.jpg'))
    # demo(sess, net, '0_Parade_marchingband_1_333.jpg')
    # # # plt.show()



