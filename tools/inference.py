#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: inference.py
@time: 18-7-19 下午6:01
@desc:
'''
import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np
import cv2
import os
from model.config import cfg

input_tensor = tf.placeholder(tf.float32, shape=[1, None, None, 3])

PATH_TO_PB = '/home/oeasy/PycharmProjects/deep-head-pose-tf/output/mobile/Pose_300W_LP/default_alpha_0.01/pb_ckpt_1.2/mobilev1_headpose.pb'
# PATH_TO_CKPT = '/home/oeasy/PycharmProjects/tf-ssh_modify/output/vgg16/wider_face_train/pb_ckpt_1.4/vgg16_ssh_three_branches.pb'

def load_model():
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(PATH_TO_PB)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

def init_headpose_network():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    with sess.as_default():
        load_model()
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {
            output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['pose_output']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        # im_info_tensor = tf.get_default_graph().get_tensor_by_name('im_info:0')
        return lambda img: sess.run(tensor_dict,
                                                 feed_dict={image_tensor: img})

def run_inference_for_one_image(image):
    im = cv2.resize(image, (227, 227))
    # im = im[np.newaxis, :, :, :]
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im = im_orig[np.newaxis, :, :, :]
    result_dict = model_headpose(im)
    return result_dict

if __name__ == '__main__':
    # detection_graph = import_graph()
    # im_names = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    model_headpose = init_headpose_network()
    # for im_name in im_names:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for {}'.format(im_name))
    #     # im = cv2.imread(os.path.join(PATH_TO_TEST_IMAGES_DIR, im_name))
    #     im = cv2.imread(os.path.join(PATH_TO_TEST_IMAGES_DIR, '4c059278d0f9254d0b077df3473755f83bb65cb3.jpg'))
    #     print('>>>>>', im.shape[0], im.shape[1])
    #     timer = Timer()
    #     timer.tic()
    #     boxes = run_inference_for_one_image(im)
    #     timer.toc()
    #     print('Detection took {:.3f}s'.format(timer.total_time))
    #     print('>>>>>faces:', boxes.shape[0])


    # run_inference_for_images(PATH_TO_TEST_IMAGES_DIR, im_names, detection_graph)

    # print('>>>>', im_names)
    # for im_name in im_names:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for {}'.format(im_name))
    #     im = cv2.imread(os.path.join(PATH_TO_TEST_IMAGES_DIR, im_name))
    #     print('>>>>>', im.shape[0], im.shape[1])
    #     boxes = run_inference_for_single_image(im, detection_graph)
    #     print('>>>>>faces', boxes.shape[0])

    ########################## write annotation txt file #############################################
    model_predtxt = open('../mobile_Pose_300W_LP_test_AFLW2000_headpose_alpha_0.01', 'w')
    # data_path = '/home/oeasy/Downloads/dataset/head_pose/AFLW2000_headpose'
    PATH_TO_TEST_IMAGES_DIR = '/home/oeasy/Downloads/dataset/head_pose/AFLW2000_headpose'
    im_names = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    print('>>>>>', im_names)
    while True:
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            img_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, im_name)
            im = cv2.imread(img_path)
            print('Demo for {}'.format(img_path))
            result = run_inference_for_one_image(im)
            yaw = result['pose_output'][0]
            pitch = result['pose_output'][1]
            roll = result['pose_output'][2]
            print('>>>>>', yaw, pitch, roll)
            txt = im_name + ',' + str(yaw) + ',' + str(pitch) + ',' + str(roll) + '\n'
            model_predtxt.write(txt)
        model_predtxt.close()





