from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Epoch size for reducing the learning rate, currently only support one step
__C.TRAIN.EPOCHSIZE = [0]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
# __C.TRAIN.DOUBLE_BIAS = True
__C.TRAIN.DOUBLE_BIAS = False

# the coefficient of regression loss
__C.TRAIN.ALPHA = 0.001

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = True

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 6

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 32

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Iterations between snapshots
# __C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.SNAPSHOT_EPOCHS = 5

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
# __C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
__C.TRAIN.SNAPSHOT_PREFIX = 'vgg16_hopenet'


#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the bottom 5 of 14 layers is fixed
# Range: 0 (none) to 12 (all)
# __C.MOBILENET.FIXED_LAYERS = 5
__C.MOBILENET.FIXED_LAYERS = 0

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1


#
# MobileNet_V2 options
#

__C.MOBILENET_V2 = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET_V2.REGU_DEPTH = False

# Number of fixed layers during training, by default the bottom 5 of 14 layers is fixed
# Range: 0 (none) to 12 (all)
# __C.MOBILENET.FIXED_LAYERS = 5
__C.MOBILENET_V2.FIXED_LAYERS = 0

# Weight decay for the mobilenet weights
__C.MOBILENET_V2.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET_V2.DEPTH_MULTIPLIER = 1.

# min_depth
__C.MOBILENET_V2.MIN_DEPTH = 16


#
# MISC
#

#
# Darknet53 options
#
__C.DARKNET = edict()

# Weight decay for the mobilenet weights
__C.DARKNET.WEIGHT_DECAY = 0.00004

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'


def get_output_dir(name, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def get_output_tb_dir(name, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    print("^^^^^^^^^^^:", cfg_list[0::2], cfg_list[1::2])
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # print("%%%%%%%%%%:", k)
        key_list = k.split('.')
        # print("&&&&&&&&&&&:", key_list)
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        # assert type(value) == type(d[subkey]), \
        #   'type {} does not match original type {}'.format(
        #     type(value), type(d[subkey]))
        d[subkey] = value
