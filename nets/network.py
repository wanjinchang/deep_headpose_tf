#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: PyCharm
@file: network.py
@time: 18-6-22 上午9:38
@desc: modify from Xinlei Chen
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tools._init_paths
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from model.config import cfg
from utils.visualization import draw_axis
from utils.timer import Timer

class Network(object):
    def __init__(self):
        self._predictions = {}
        self._all_preds = {}
        self._losses = {}
        self._all_losses = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        # resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        # plot only the first image of one batch
        self._gt_image = tf.reverse(image[0, :, :, :], axis=[-1])
        self._gt_image = tf.expand_dims(self._gt_image, 0)

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_axis, [self._gt_image, self._labels_cont[0][0], self._labels_cont[0][1],
                                     self._labels_cont[0][2]], tf.float32, name="gt_pose")
        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        else:
            initializer = slim.xavier_initializer_conv2d(uniform=True)

        timer = Timer()
        timer.tic()
        net_conv = self._image_to_head(is_training)
        timer.toc()
        print('base_network took {:.3f}s'.format(timer.total_time))
        with tf.variable_scope(self._scope, self._scope):
            fc_flatten = slim.flatten(net_conv, scope='flatten')
            yaw_pred = slim.fully_connected(fc_flatten, self._num_bins,
                                                 weights_initializer=initializer,
                                                 trainable=is_training,
                                                 activation_fn=None, scope='yaw_fc')
            pitch_pred = slim.fully_connected(fc_flatten, self._num_bins,
                                                 weights_initializer=initializer,
                                                 trainable=is_training,
                                                 activation_fn=None, scope='pitch_fc')
            roll_pred = slim.fully_connected(fc_flatten, self._num_bins,
                                                 weights_initializer=initializer,
                                                 trainable=is_training,
                                                 activation_fn=None, scope='roll_fc')
            self._predictions['yaw'] = yaw_pred
            self._predictions['pitch'] = pitch_pred
            self._predictions['roll'] = roll_pred
            self._score_summaries.update(self._predictions)

    def _compute_mse_loss(self):
        raise NotImplementedError

    def _compute_losses(self):
        self._losses['total_loss'] = 0
        for k, axis in enumerate(['yaw', 'pitch', 'roll']):
            loss = self._add_losses(k, axis)
            self._losses['total_loss'] += loss
        total_loss = self._losses['total_loss']
        self._all_losses['all_losses'] = self._losses
        self._event_summaries.update(self._all_losses)
        return total_loss

    def _add_losses(self, k, axis):
        with tf.variable_scope('LOSS_' + axis + '_' + self._tag) as scope:
            # Cross Entropy loss
            loss_items = {}
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._predictions[axis],
                                                               labels=self._labels[:, k]))
            loss_items['cross_entropy'] = cross_entropy_loss

            # MSE loss
            pred = tf.nn.softmax(self._predictions[axis], name=axis + '_pred')
            pred = tf.reduce_sum(pred * self._idx_tensor, 1) * tf.constant(self._interval) - tf.constant(99)
            reg_loss = tf.losses.mean_squared_error(labels=self._labels_cont[:, k], predictions=pred)
            loss_items['mse'] = reg_loss

            # total loss
            total_loss = cross_entropy_loss + cfg.TRAIN.ALPHA * reg_loss
            loss_items['total_loss'] = total_loss
            self._losses[axis] = loss_items
        return total_loss

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def create_architecture(self, mode, interval, num_bins, tag=None):
        self._image = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='image_tensor')
        # Binned labels placeholder
        self._labels = tf.placeholder(tf.int64, shape=[None, 3], name='labels')

        # Continuous labels placeholder
        self._labels_cont = tf.placeholder(tf.float32, shape=[None, 3], name='labels_cont')

        self._tag = tag
        self._interval = interval
        self._num_bins = num_bins
        self._idx_tensor = tf.range(0, self._num_bins, dtype=tf.float32, name='range')

        self._mode = mode

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self._build_network(training)

        # layers_to_output = {'rois': rois}
        layers_to_output = {}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if training:
            # self._add_losses()
            self._compute_losses()
            layers_to_output.update(self._all_losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    for k1, v1 in var.items():
                        if isinstance(v1, dict):
                            for k2, v2 in v1.items():
                                val_summaries.append(tf.summary.scalar(key + '_' + k1 + '_' + k2, v2))
                        else:
                            val_summaries.append(tf.summary.scalar(key + '_' + k1, v1))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                # for var in self._train_summaries:
                #     self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            # self._summary_op_val = tf.summary.merge(val_summaries)
        else:
            yaw = tf.nn.softmax(self._predictions['yaw'], name='yaw_pred')
            pitch = tf.nn.softmax(self._predictions['pitch'], name='pitch_pred')
            roll = tf.nn.softmax(self._predictions['roll'], name='roll_pred')

            # Continuous predictions
            yaw_pred = tf.reduce_sum(yaw * self._idx_tensor, 1) * tf.constant(self._interval, dtype=tf.float32) - tf.constant(99, dtype=tf.float32)
            pitch_pred = tf.reduce_sum(pitch * self._idx_tensor, 1) * tf.constant(self._interval, dtype=tf.float32) - tf.constant(99, dtype=tf.float32)
            roll_pred = tf.reduce_sum(roll * self._idx_tensor, 1) * tf.constant(self._interval, dtype=tf.float32) - tf.constant(99, dtype=tf.float32)

            pose_output = tf.concat([yaw_pred, pitch_pred, roll_pred], name='pose_output', axis=0)
            self._predictions['yaw_raw'] = yaw
            self._predictions['pitch_raw'] = pitch
            self._predictions['roll_raw'] = roll
            self._predictions['yaw_pred'] = yaw_pred
            self._predictions['pitch_pred'] = pitch_pred
            self._predictions['roll_pred'] = roll_pred
            self._predictions['pose_output'] = pose_output

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # only useful during testing mode
    def test_image(self, sess, image):
        feed_dict = {self._image: image}
        timer = Timer()
        timer.tic()
        predictions = sess.run(self._predictions, feed_dict=feed_dict)
        timer.toc()
        print('Prediction took {:.3f}s'.format(timer.total_time))
        yaw_raw = predictions['yaw_raw']
        pitch_raw = predictions['pitch_raw']
        roll_raw = predictions['roll_raw']
        yaw = predictions['pose_output'][0]
        pitch = predictions['pose_output'][1]
        roll = predictions['pose_output'][2]
        return yaw, pitch, roll, yaw_raw, pitch_raw, roll_raw

    def get_summary(self, sess, images_batch, labels_batch, labels_cont_batch):
        feed_dict = {self._image: images_batch, self._labels: labels_batch, self._labels_cont: labels_cont_batch}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
        return summary

    def train_step_old(self, sess, images_batch, labels_batch, labels_cont_batch, train_op, module):
        feed_dict = {self._image: images_batch, self._labels: labels_batch, self._labels_cont: labels_cont_batch}
        cls_loss, reg_loss, loss, _ = sess.run([self._losses[module]["cross_entropy"],
                                                        self._losses[module]['mse'],
                                                        self._losses[module]['total_loss'],
                                                        train_op],
                                                       feed_dict=feed_dict)
        return cls_loss, reg_loss, loss

    def train_step(self, sess, images_batch, labels_batch, labels_cont_batch, train_op):
        feed_dict = {self._image: images_batch, self._labels: labels_batch, self._labels_cont: labels_cont_batch}
        losses, _ = sess.run([self._losses, train_op], feed_dict=feed_dict)
        return losses

    def train_step_with_summary_old(self, sess, images_batch, labels_batch, labels_cont_batch, train_op, module):
        feed_dict = {self._image: images_batch, self._labels: labels_batch, self._labels_cont: labels_cont_batch}
        cls_loss, reg_loss, loss, summary, _ = sess.run([self._losses[module]["cross_entropy"],
                                                                     self._losses[module]['mse'],
                                                                     self._losses['total_loss'],
                                                                     self._summary_op,
                                                                     train_op],
                                                                    feed_dict=feed_dict)
        return  cls_loss, reg_loss, loss, summary

    def train_step_with_summary(self, sess, images_batch, labels_batch, labels_cont_batch, train_op):
        feed_dict = {self._image: images_batch, self._labels: labels_batch, self._labels_cont: labels_cont_batch}
        losses, summary, _ = sess.run([self._losses, self._summary_op, train_op], feed_dict=feed_dict)
        return losses, summary

    def train_step_no_return(self, sess, images_batch, labels_batch, labels_cont_batch, train_op):
        feed_dict = {self._image: images_batch, self._labels: labels_batch, self._labels_cont: labels_cont_batch}
        sess.run([train_op], feed_dict=feed_dict)
