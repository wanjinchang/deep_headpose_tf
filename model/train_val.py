# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from model.config import cfg
from utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import glob
import time

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

class SolverWrapper(object):
    """
      A wrapper class for the training process
    """

    def __init__(self, sess, network, output_dir, tbdir, pretrained_model=None):
        self.net = network
        # self.imdb = imdb
        # self.roidb = roidb
        # self.valroidb = valroidb
        self.output_dir = output_dir
        self.tbdir = tbdir
        # Simply put '_val' at the end to save the summaries from the validation set
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        self.pretrained_model = pretrained_model

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_epoch_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_epoch_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            last_snapshot_epoch = pickle.load(fid)

        return last_snapshot_epoch

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            # reader = tf.train.NewCheckpointReader(file_name)
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def construct_graph(self, sess, interval, num_bins):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', interval, num_bins, tag='default')

            losses = layers['all_losses']
            total_loss = losses['total_loss']
            yaw_loss = losses['yaw']['total_loss']
            pitch_loss = losses['pitch']['total_loss']
            roll_loss = losses['roll']['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            self.optimizer_yaw = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            self.optimizer_pitch = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
            self.optimizer_roll = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # Batch norm requires update_ops to be added as a train_op dependency.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Set learning rate use Adam optimizer

                # Compute the gradients with regard to the loss
                gvs = self.optimizer.compute_gradients(total_loss)
                gvs_m1 = self.optimizer_yaw.compute_gradients(yaw_loss)
                gvs_m2 = self.optimizer_pitch.compute_gradients(pitch_loss)
                gvs_m3 = self.optimizer_roll.compute_gradients(roll_loss)
                # gvs_m1 = self.optimizer.compute_gradients(m1_loss)
                # gvs_m2 = self.optimizer.compute_gradients(m2_loss)
                # gvs_m3 = self.optimizer.compute_gradients(m3_loss)
                # Double the gradient of the bias if set
                if cfg.TRAIN.DOUBLE_BIAS:
                    final_gvs = []
                    final_gvs_m1 = []
                    final_gvs_m2 = []
                    final_gvs_m3 = []
                    with tf.variable_scope('Gradient_Mult') as scope:
                        for grad, var in gvs:
                            # print("grad, var:", grad, var)
                            scale = 1.
                            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                                scale *= 2.
                            if not np.allclose(scale, 1.0):
                                grad = tf.multiply(grad, scale)
                            final_gvs.append((grad, var))
                    train_op = self.optimizer.apply_gradients(final_gvs)
                    with tf.variable_scope('Gradient_Mult_yaw') as scope:
                        for grad, var in gvs_m1:
                            scale = 1.
                            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                                scale *= 2.
                            if not np.allclose(scale, 1.0):
                                grad = tf.multiply(grad, scale)
                            final_gvs_m1.append((grad, var))
                    train_yaw_op = self.optimizer.apply_gradients(final_gvs_m1)
                    with tf.variable_scope('Gradient_Mult_pitch') as scope:
                        for grad, var in gvs_m2:
                            scale = 1.
                            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                                scale *= 2.
                            if not np.allclose(scale, 1.0):
                                grad = tf.multiply(grad, scale)
                            final_gvs_m2.append((grad, var))
                    train_pitch_op = self.optimizer.apply_gradients(final_gvs_m2)
                    with tf.variable_scope('Gradient_Mult_roll') as scope:
                        for grad, var in gvs_m3:
                            scale = 1.
                            if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                                scale *= 2.
                            if not np.allclose(scale, 1.0):
                                grad = tf.multiply(grad, scale)
                            final_gvs_m3.append((grad, var))
                    train_roll_op = self.optimizer.apply_gradients(final_gvs_m3)
                    final_train_op = tf.group(train_yaw_op, train_pitch_op, train_roll_op)
                else:
                    train_op = self.optimizer.apply_gradients(gvs)
                    train_yaw_op = self.optimizer_yaw.apply_gradients(gvs_m1)
                    train_pitch_op = self.optimizer_pitch.apply_gradients(gvs_m2)
                    train_roll_op = self.optimizer_roll.apply_gradients(gvs_m3)
                    final_train_op = tf.group(train_yaw_op, train_pitch_op, train_roll_op)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            # self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return lr, train_op, train_yaw_op, train_pitch_op, train_roll_op, final_train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_epoch_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for epochsize in cfg.TRAIN.EPOCHSIZE:
            redfiles.append(os.path.join(self.output_dir,
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_epoch_{:d}.ckpt.meta'.format(epochsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_epoch_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        print("variables:", variables)
        if self.pretrained_model == '':
            print('Training from Scratch!!!')
            init = tf.global_variables_initializer()
            sess.run(init)
            last_snapshot_epoch = 0
            rate = cfg.TRAIN.LEARNING_RATE
            epochsizes = list(cfg.TRAIN.EPOCHSIZE)
        elif 'darknet53' in self.pretrained_model:
            print('The base network is Darknet53!!!')
            sess.run(tf.variables_initializer(variables, name='init'))
            self.net.restored_from_npz(sess)
            print('Loaded.')
            last_snapshot_epoch = 0
            rate = cfg.TRAIN.LEARNING_RATE
            epochsizes = list(cfg.TRAIN.EPOCHSIZE)
        elif 'squeezenetv1' in self.pretrained_model:
            print('The base network is SqueezeNetV1 !!!')
            sess.run(tf.variables_initializer(variables, name='init'))
            self.net.restored_from_mat(sess, self.pretrained_model)
            print('Loaded.')
            # print('>>>>>>>', variables[0].eval())
            last_snapshot_epoch = 0
            rate = cfg.TRAIN.LEARNING_RATE
            epochsizes = list(cfg.TRAIN.EPOCHSIZE)
        else:
            sess.run(tf.variables_initializer(variables, name='init'))
            var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
            # print("var_keep_dic", var_keep_dic)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            print('Loaded.')

            # Need to fix the variables before loading, so that the RGB weights are changed to BGR
            # For VGG16 it also changes the convolutional weights
            self.net.fix_variables(sess, self.pretrained_model)
            print('Fixed.')
            last_snapshot_epoch = 0
            rate = cfg.TRAIN.LEARNING_RATE
            epochsizes = list(cfg.TRAIN.EPOCHSIZE)

        return rate, last_snapshot_epoch, epochsizes, np_paths, ss_paths

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        variables = tf.global_variables()
        print("variables:", variables)
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_epoch = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        rate = cfg.TRAIN.LEARNING_RATE
        epochsizes = []
        for epochsize in cfg.TRAIN.EPOCHSIZE:
            if last_snapshot_epoch > epochsize:
                rate *= cfg.TRAIN.GAMMA
            else:
                epochsizes.append(epochsize)

        return rate, last_snapshot_epoch, epochsizes, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def train_model(self, sess, data_class, max_epochs, interval, num_bins):
        # Construct the computation graph
        lr, train_op, train_yaw_op, train_pitch_op, train_roll_op, final_train_op = self.construct_graph(sess, interval, num_bins)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_epoch, epochsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_epoch, epochsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                   str(nfiles[-1]))
        timer = Timer()
        epoch = last_snapshot_epoch + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        epochsizes.append(max_epochs)
        epochsizes.reverse()
        next_epochsize = epochsizes.pop()
        sess.run(tf.assign(lr, rate))
        while epoch < max_epochs + 1:
            train_gen = data_class._next_batch()
            num_batches_train = int(data_class.length // data_class.batch_size)
            for iter in range(num_batches_train):
                if epoch == next_epochsize + 1:
                    # Add snapshot here before reducing the learning rate
                    self.snapshot(sess, epoch)
                    rate *= cfg.TRAIN.GAMMA
                    sess.run(tf.assign(lr, rate))
                    next_epochsize = epochsizes.pop()

                timer.tic()
                # Get training data, one batch at a time
                images_batch, labels_batch, labels_cont_batch = next(train_gen)
                now = time.time()
                if iter == 0 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                    # Compute the graph with summary
                    losses, summary = self.net.train_step_with_summary(sess, images_batch,
                                                            labels_batch, labels_cont_batch, final_train_op)
                    # self.writer.add_summary(summary, float((epoch-1) * num_batches_train + iter))
                    self.writer.add_summary(summary, float((epoch-1) * num_batches_train + iter))
                    # # Also check the summary on the validation set
                    # blobs_val = self.data_layer_val.forward()
                    # summary_val = self.net.get_summary(sess, blobs_val)
                    # self.valwriter.add_summary(summary_val, float(iter))
                    last_summary_time = now
                else:
                    # Compute the graph without summary
                    losses = self.net.train_step(sess, images_batch,  labels_batch, labels_cont_batch, final_train_op)
                total_loss = losses['total_loss']

                yaw_cross_entropy = losses['yaw']['cross_entropy']
                yaw_mse = losses['yaw']['mse']
                yaw_total_loss = losses['yaw']['total_loss']

                pitch_cross_entropy = losses['pitch']['cross_entropy']
                pitch_mse = losses['pitch']['mse']
                pitch_total_loss = losses['pitch']['total_loss']

                roll_cross_entropy = losses['roll']['cross_entropy']
                roll_mse = losses['roll']['mse']
                roll_total_loss = losses['roll']['total_loss']

                timer.toc()

                # Display training information
                if iter % (cfg.TRAIN.DISPLAY) == 0:
                    print('epoch: %d / %d,  iter: %d / %d, \n >>> yaw_cls_loss: %.6f, yaw_reg_loss: %.6f, yaw_total loss: %.6f\n '
                          '>>> pitch_cls_loss: %.6f, pitch_reg_loss: %.6f, pitch_total loss: %.6f\n '
                          '>>> roll_cls_loss: %.6f, roll_reg_loss: %.6f, roll_total loss: %.6f\n '
                          '>>> total_loss: %.6f, lr: %f'% \
                          (epoch, max_epochs, iter, num_batches_train, yaw_cross_entropy, yaw_mse, yaw_total_loss,
                           pitch_cross_entropy, pitch_mse, pitch_total_loss, roll_cross_entropy, roll_mse, roll_total_loss, total_loss,
                           lr.eval()))
                    print('speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if epoch % cfg.TRAIN.SNAPSHOT_EPOCHS == 0:
                last_snapshot_epoch = epoch
                ss_path, np_path = self.snapshot(sess, epoch)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            epoch += 1

        if last_snapshot_epoch != epoch - 1:
            self.snapshot(sess, epoch - 1)

        self.writer.close()
        # self.valwriter.close()

    def train_model_old(self, sess, data_class, max_epochs, num_bins):
        # Construct the computation graph
        # lr, train_op = self.construct_graph(sess)
        lr, train_op, train_yaw_op, train_pitch_op, train_roll_op, _ = self.construct_graph(sess, interval, num_bins)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_epoch, epochsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_epoch, epochsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                   str(nfiles[-1]))
        timer = Timer()
        epoch = last_snapshot_epoch + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        epochsizes.append(max_epochs)
        epochsizes.reverse()
        next_epochsize = epochsizes.pop()
        while epoch < max_epochs + 1:
            train_gen = data_class._next_batch()
            num_batches_train = int(data_class.length // data_class.batch_size)
            for iter in range(num_batches_train):
                random_seed = np.random.rand()
                if random_seed < 0.33:
                    module = 'yaw'
                    train_op = train_yaw_op
                elif 0.33 <= random_seed < 0.67:
                    module = 'pitch'
                    train_op = train_pitch_op
                else:
                    module = 'roll'
                    train_op = train_roll_op
                # Learning rate
                if epoch == next_epochsize + 1:
                    # Add snapshot here before reducing the learning rate
                    self.snapshot(sess, epoch)
                    rate *= cfg.TRAIN.GAMMA
                    sess.run(tf.assign(lr, rate))
                    next_epochsize = epochsizes.pop()

                timer.tic()
                # Get training data, one batch at a time
                images_batch, labels_batch, labels_cont_batch = next(train_gen)
                now = time.time()
                if iter == 0 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                    # Compute the graph with summary
                    cross_entropy, reg_loss, total_loss, summary = self.net.train_step_with_summary_old(sess, images_batch,
                                                            labels_batch, labels_cont_batch, train_op, module)
                    self.writer.add_summary(summary, float((epoch-1) * num_batches_train + iter))
                    # # Also check the summary on the validation set
                    # blobs_val = self.data_layer_val.forward()
                    # summary_val = self.net.get_summary(sess, blobs_val)
                    # self.valwriter.add_summary(summary_val, float(iter))
                    last_summary_time = now
                else:
                    # Compute the graph without summary
                    cross_entropy, reg_loss, total_loss = self.net.train_step_old(sess, images_batch,  labels_batch,
                                                                         labels_cont_batch, train_op, module)
                timer.toc()

                # Display training information
                if iter % (cfg.TRAIN.DISPLAY) == 0:
                    print('epoch: %d / %d,  iter: %d / %d, now training module: %s\n >>> total loss: %.6f\n >>> cls_loss: %.6f\n '
                          '>>> reg_loss: %.6f\n >>> lr: %f' % \
                          (epoch, max_epochs, iter, num_batches_train, module, total_loss, cross_entropy, reg_loss, lr.eval()))
                    print(' speed: {:.3f}s / iter'.format(timer.average_time))

            # Snapshotting
            if epoch % cfg.TRAIN.SNAPSHOT_EPOCHS == 0:
                last_snapshot_epoch = epoch
                ss_path, np_path = self.snapshot(sess, epoch)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)
            epoch += 1

        if last_snapshot_epoch != epoch - 1:
            self.snapshot(sess, epoch - 1)

        self.writer.close()
        # self.valwriter.close()


def train_net(network, data_class, output_dir, tb_dir,
              pretrained_model=None,
              max_epochs=400, interval=3, num_bins=66):
    """Train a Hopenet network."""

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, output_dir, tb_dir,
                           pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, data_class, max_epochs, interval, num_bins)
        print('done solving')
