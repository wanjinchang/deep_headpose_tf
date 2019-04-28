from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops

from nets.network import Network
import scipy.io
import os
import numpy as np
import time

# squeezenet_v1.1
# https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/deploy.prototxt
sqz_prefix = ['conv1', 'fire2', 'fire3', 'fire4', 'fire5', 'fire6', 'fire7', 'fire8', 'fire9']

def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)

def squeezenet_v1_base(inputs, keep_probability=0.6, bottleneck_layer_size=128,
              reuse=None, base_only=True, scope=None):
    with tf.variable_scope(scope, 'SqueezenetV1', [inputs], reuse=reuse):
        net = slim.conv2d(inputs, 64, [3, 3], stride=2, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net = fire_module(net, 16, 64, scope='fire3')
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire4')
        net = fire_module(net, 32, 128, scope='fire5')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
        net = fire_module(net, 48, 192, scope='fire6')
        net = fire_module(net, 48, 192, scope='fire7')
        net = fire_module(net, 64, 256, scope='fire8')
        net = fire_module(net, 64, 256, scope='fire9')
        if base_only:
            return net, None
        net = slim.dropout(net, keep_probability)
        net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
        net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
        net = tf.squeeze(net, [1, 2], name='logits')
        net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                   scope='Bottleneck', reuse=False)
        return net, None

def squeezenetv1_arg_scope(is_training=True, weight_decay=0.0):
    weights_initializer = slim.xavier_initializer_conv2d(uniform=True)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        trainable=is_training,
                        weights_initializer=weights_initializer,
                        weights_regularizer=regularizer,
                        biases_initializer=tf.constant_initializer(0.0),
                        activation_fn=tf.nn.relu,
                        padding='SAME') as sc:
        return sc

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0,
              reuse=None, base_only=True):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    weights_init = tf.truncated_normal_initializer(stddev=0.09)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=weights_init,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=init_ops.zeros_initializer(),
                        activation_fn=tf.nn.relu,
                        trainable=phase_train,
                        normalizer_fn=slim.batch_norm):
        with tf.variable_scope('SqueezenetV1', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=phase_train):
                    net = slim.conv2d(images, 64, [3, 3], stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
                    net = fire_module(net, 16, 64, scope='fire2')
                    net = fire_module(net, 16, 64, scope='fire3')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
                    net = fire_module(net, 32, 128, scope='fire4')
                    net = fire_module(net, 32, 128, scope='fire5')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
                    net = fire_module(net, 48, 192, scope='fire6')
                    net = fire_module(net, 48, 192, scope='fire7')
                    net = fire_module(net, 64, 256, scope='fire8')
                    net = fire_module(net, 64, 256, scope='fire9')
                    if base_only:
                        return net, None
                    net = slim.dropout(net, keep_probability)
                    net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
                    net = tf.squeeze(net, [1, 2], name='logits')
                    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                            scope='Bottleneck', reuse=False)
    return net, None

class squeezenet_v1(Network):
    def __init__(self):
        Network.__init__(self)
        self._scope = 'SqueezenetV1'
        # self.sqz_mat_path = sqz_backonemat_path

    def _image_to_head(self, is_training, reuse=None):
        # Base bottleneck
        net_conv = self._image
        # with slim.arg_scope(squeezenetv1_arg_scope(is_training=is_training)):
        #     net_conv, _ = squeezenet_v1_base(net_conv, reuse=reuse, scope=self._scope)
        net_conv, _ = inference(net_conv, keep_probability=0.6, phase_train=is_training)
        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv

        return net_conv

    def get_dtype_np(self):
        return np.float32

    def get_dtype_tf(self):
        return tf.float32

    def get_weights_biases(self, preloaded, layer_name):
        weights, biases = preloaded[layer_name]
        biases = biases.reshape(-1)
        return (weights, biases)

    def load_net(self, data_path):
        if not os.path.isfile(data_path):
            print("Network %s does not exist. (Did you forget to download it?)" % data_path)

        weights_raw = scipy.io.loadmat(data_path)

        # Converting to needed type
        conv_time = time.time()
        weights = {}
        for name in weights_raw:
            # skipping '__version__', '__header__', '__globals__'
            if name[0:2] != '__':
                kernels, bias = weights_raw[name][0]
                weights[name] = []
                weights[name].append(kernels.astype(self.get_dtype_np()))
                weights[name].append(bias.astype(self.get_dtype_np()))
        print("Converted network data(%s): %fs" % (self.get_dtype_np(), time.time() - conv_time))

        # mean_pixel = np.array([104.006, 116.669, 122.679], dtype=self.get_dtype_np())
        return weights

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv_0/conv_weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fire_cluster(self, sess, preloaded, cluster_name):
        # central -> squeeze
        layer_name = cluster_name + '/squeeze1x1'
        weights, biases = self.get_weights_biases(preloaded, layer_name)
        sess.run(tf.get_variable(cluster_name + '/squeeze/weights').assign(weights))
        sess.run(tf.get_variable(cluster_name + '/squeeze/biases').assign(biases))

        # left - expand 1x1
        layer_name = cluster_name + '/expand1x1'
        weights, biases = self.get_weights_biases(preloaded, layer_name)
        sess.run(tf.get_variable(cluster_name + '/expand/1x1/weights').assign(weights))
        sess.run(tf.get_variable(cluster_name + '/expand/1x1/biases').assign(biases))

        # right - expand 3x3
        layer_name = cluster_name + '/expand3x3'
        weights, biases = self.get_weights_biases(preloaded, layer_name)
        sess.run(tf.get_variable(cluster_name + '/expand/3x3/weights').assign(weights))
        sess.run(tf.get_variable(cluster_name + '/expand/3x3/biases').assign(biases))

    def restored_from_mat(self, sess, pretrained_model):
        if pretrained_model != '':
            # data_dict = np.load(self.darknet53_npz_path)
            preloaded = self.load_net(pretrained_model)
            data_dict = self.fix_first_conv(preloaded)
        else:
            print('the squeezenet model does not exist!!!')
            return
        print('restored variables from squeezenetv1_backbone_mat!!!!')
        with tf.variable_scope(self._scope, reuse=True):
            for key in sqz_prefix:
                print('restoring layer {}'.format(key))
                if key == 'conv1':
                    weights, biases = self.get_weights_biases(data_dict, key)
                    sess.run(tf.get_variable(key + '/weights').assign(weights))
                    sess.run(tf.get_variable(key + '/biases').assign(biases))
                else:
                    self.fire_cluster(sess, preloaded, key)

    def fix_first_conv(self, preloaded):
        # fix the first conv layer channel from RGB to BGR
        print('Fix squeezenet first conv layers..')
        weights, biases = preloaded['conv1']
        # biases = biases.reshape(-1)
        preloaded['conv1'][0] = weights[:, :, ::-1, :]
        preloaded['conv1'][1] = biases
        return preloaded




