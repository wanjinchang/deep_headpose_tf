import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
from utils import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class Pose_300W_LP():
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        # img = img.convert(self.image_mode)
        img_path = self.X_train[index]
        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        # k = np.random.random_sample() * 0.2 + 0.2
        k = np.random.random_sample() * 0.15 + 0.15
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        img = img[int(y_min):int(y_max), int(x_min):int(x_max), :]

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = img.filter(ImageFilter.BLUR)
        #
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, img_path, yaw, pitch, roll

class AFLW2000():
    def __init__(self, data_dir, filename_path, transform=None, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        # img = img.convert(self.image_mode)
        # mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_path = self.X_train[index]
        img = cv2.imread(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.15
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        # if x_min < 0:
        #     x_min = 0
        # if y_min < 0:
        #     y_min = 0
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        face = img[abs(int(y_min)):abs(int(y_max)), abs(int(x_min)):abs(int(x_max))]

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        return face, img_path, yaw, pitch, roll

    def __len__(self):
        # 2,000
        return self.length


if __name__ == '__main__':
    ################################ 300W_LP Dataset ###########################################
    anno_txt = open('300W_LP_headpose_anno.txt', 'w')
    data_dir = '/home/oeasy/Downloads/dataset/head_pose/300W_LP'
    face_data_path = '/home/oeasy/Downloads/dataset/head_pose/300W_LP_headpose'
    filename_path = '/home/oeasy/Downloads/dataset/head_pose/300W_LP/300W_LP_anno.txt'
    pose_dataset = Pose_300W_LP(data_dir=data_dir, filename_path=filename_path, transform=None)
    img_num = 0
    while True:
        for data in pose_dataset:
            img_num += 1
            img = data[0]
            img_name = data[1].split('/')[1] + ('_%s.jpg' % img_num)
            img_path = os.path.join(face_data_path, img_name)
            print('writing {} for {}th image'.format(img_name, img_num))
            yaw = data[2]
            pitch = data[3]
            roll = data[4]
            # ignore that over 99 degree
            if abs(yaw) > 99 or abs(pitch) > 99 or abs(roll) > 99:
                continue
            cv2.imwrite(img_path, img)
            txt = img_name + ',' + str(yaw) + ',' + str(pitch) + ',' + str(roll) + '\n'
            anno_txt.write(txt)
        print('writing total {} images...'.format(img_num))
        anno_txt.close()

    ################################# AFLW2000 Dataset ####################################
    # anno_txt = open('AFLW2000_headpose_anno.txt', 'w')
    # data_dir = '/home/oeasy/Downloads/dataset/head_pose/AFLW2000'
    # face_data_path = '/home/oeasy/Downloads/dataset/head_pose/AFLW2000_headpose'
    # filename_path = '/home/oeasy/Downloads/dataset/head_pose/AFLW2000_anno.txt'
    # pose_dataset = AFLW2000(data_dir=data_dir, filename_path=filename_path, transform=None)
    # img_num = 0
    # while True:
    #     for data in pose_dataset:
    #         img_num += 1
    #         face = data[0]
    #         img_name = data[1].strip() + '.jpg'
    #         img_path = os.path.join(face_data_path, img_name)
    #         print('>>>>>path', img_path)
    #         print('writing {} for {}th image'.format(img_name, img_num))
    #         print('>>>>>shape', face.shape)
    #         cv2.imwrite(img_path, face)
    #         yaw = data[2]
    #         pitch = data[3]
    #         roll = data[4]
    #         txt = img_name + ',' + str(yaw) + ',' + str(pitch) + ',' + str(roll) + '\n'
    #         anno_txt.write(txt)
    #     print('writing total {} images...'.format(img_num))
    #     anno_txt.close()

