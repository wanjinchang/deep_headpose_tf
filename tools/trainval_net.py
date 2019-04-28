# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import pprint
import numpy as np
f_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f_path + '/..')

# import tools._init_paths
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from model.train_val import train_net
from utils.datasets import Pose_300W_LP_DataSet

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from nets.mobilenet_v2.mobilenet_v2 import mobilenetv2
from nets.squeezenet_v1_1 import squeezenet_v1

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Hopenet network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/squeezenetv1.yml', type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights', default='data/imagenet_weights/squeezenetv1_backbone.mat',
                        type=str)
    parser.add_argument('--gpu_id', dest='gpu_id', help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=32, type=int)
    parser.add_argument('--interval', dest='interval', help='degree interval.',
                        default=3, type=int)
    parser.add_argument('--num_bins', dest='num_bins', help='Num bins(value=(99-(-99))/interval).',
                        default=66, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to train on',
                        default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
                        default='/home/oeasy/Downloads/dataset/head_pose/headpose_data/oeasy_headpose', type=str)
    parser.add_argument('--annotation_path', dest='annotation_path',
                        help='Path to text file containing relative paths for every example.',
                        default='oeasy_headpose_anno.txt', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
                        default=0.001, type=float)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=400, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='default', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile, mobile_v2',
                        default='squeezenetv1', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # ==========================================================================
    # LIMITE THE USAGE OF THE GPU
    # =========================================================================
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set class
    pose_dataset = Pose_300W_LP_DataSet(args.data_dir, args.annotation_path, args.interval, cfg.TRAIN.BATCH_SIZE)
    print('{:d} pose entries'.format(pose_dataset.length))
    print('batch_size:{}'.format(pose_dataset.batch_size))

    # output directory where the models are saved
    output_dir = get_output_dir(args.dataset, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(args.dataset, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    elif args.net == 'mobile_v2':
        net = mobilenetv2()
    elif args.net == 'squeezenetv1':
        net = squeezenet_v1()
    else:
        raise NotImplementedError

    train_net(net, pose_dataset, output_dir, tb_dir, pretrained_model=args.weight,
              max_epochs=args.max_epochs, interval=args.interval, num_bins=args.num_bins)
