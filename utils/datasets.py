#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: datasets.py
@time: 18-8-13 下午4:44
@desc:
'''
import os
import cv2
import numpy as np
from model.config import cfg

pose_idx_dict = {"yaw": 1, "pitch": 2, "roll": 3}

######################### 300W_LP dataset loader #######################
def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.readlines()
    image_index = []
    pose_labels = []
    for line in lines:
        anno = line.strip().split(',')
        if len(anno) != 4:
            print('invalid sample {}'.format(anno))
            continue
        pose = list(map(float, anno[1:]))
        image_index.append(anno[0])
        pose_labels.append(pose)
    return lines, image_index, pose_labels

class Pose_300W_LP_DataSet():
    # Head pose Data_Generator from 300W-LP dataset
    def __init__(self, data_dir, annotation_path, interval=3, batch_size=32, target_size=(224, 224), img_ext='.jpg'):
        self.data_dir = data_dir
        self.img_ext = img_ext
        self.interval = interval
        self.batch_size = batch_size
        self.target_size = target_size

        filename_list, image_index, pose_labels = get_list_from_filenames(annotation_path)
        # print('>>>>>', filename_list)

        self.X_train = image_index
        self.Y_train = pose_labels
        # self.y_train = filename_list
        self.length = len(filename_list)

    def _next_batch(self):
        start_idx = 0
        while True:
            images_index_batch = self.X_train[start_idx:start_idx + self.batch_size]
            pose_labels_batch = self.Y_train[start_idx:start_idx + self.batch_size]
            # print('>>>>>', batch_annos)

            # generate batch image data and labels
            images = []
            labels = []
            labels_cont = []
            for i in range(self.batch_size):
                img_path = os.path.join(self.data_dir, images_index_batch[i])
                # print('>>>>>', img_path)
                img = cv2.imread(img_path)
                im_orig = img.astype(np.float32, copy=True)
                im_orig -= cfg.PIXEL_MEANS

                # image data
                # ds = 1 + np.random.randint(0, 4) * 5
                # original_size = im_orig.size
                # img = cv2.resize(im_orig, (im_orig.size[0] / ds, im_orig.size[1] / ds))
                # img = cv2.resize(img, (original_size[0], original_size[1]))
                img = cv2.resize(im_orig, (224, 224))

                yaw = pose_labels_batch[i][0]
                pitch = pose_labels_batch[i][1]
                roll = pose_labels_batch[i][2]

                # Flip?
                rnd = np.random.random_sample()
                if rnd < 0.5:
                    yaw = -yaw
                    roll = -roll
                    img = cv2.flip(img, 1)

                images.append(img)

                # Bin labels
                bins = np.array(range(-99, 102, self.interval))
                binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
                labels.append(binned_pose)

                # Cont labels
                labels_cont.append([yaw, pitch, roll])

            images = np.array(images, dtype='float32')
            labels = np.array(labels)
            labels_cont = np.array(labels_cont)

            yield (images, labels, labels_cont)

            # Update start index for the next batch
            start_idx += self.batch_size
            if start_idx >= self.length:
                start_idx = 0

if __name__ == '__main__':
    data_dir = '/home/oeasy/Downloads/dataset/head_pose/300W_LP_headpose'
    annotation_path = '../300W_LP_headpose_anno.txt'
    pose_dataset = Pose_300W_LP_DataSet(data_dir, annotation_path)

    # data_dir = '/home/oeasy/Downloads/dataset/head_pose/headpose_data/aligned_img_new'
    # annotation_path = '../img_label_new.txt'
    # pose_dataset = oeasy_headpose_DataSet(data_dir, annotation_path)
    data_gen = pose_dataset._next_batch()
    for _ in range(pose_dataset.length):
        images_batch, labels_batch, labels_cont_batch = next(data_gen)
        print('>>>>>image', images_batch.shape)
        # print('>>>>>labels', labels_batch.shape)
        print('>>>>>labels', labels_batch[0][0], labels_batch[0][1], labels_batch[0][2])
        # print('>>>>>labels', labels_batch[:, 1])
        # print('>>>>>labels_cont', labels_cont_batch[0][0], labels_cont_batch[0][1], labels_cont_batch[0][2])





