""" The T1Generator. Eventually it will take EPIs as input. """

from __future__ import division
import os
import time
from six.moves import xrange
import tensorflow as tf

import numpy as np

import ops
import imageio

class T1Generator(object):

    EPI_DIMS = [128, 128, 1] # powers of two are easier but they're actually [91, 109, 91]
    SAVE_VISUALS_EXP = 1.2
    VISUALS_DIM = [8, 8]
    CHECKPOINT_EPOCHS = 100
    STARTING_GF_DIM = 32

    def __init__(self, sess, t1_side_length=128, n_channels=1, batch_size=64,
                 checkpoint_dir='checkpoint', log_dir='log', visuals_dir='visuals'):
        """
        Args:
            sess: TensorFlow session
            t1_side_length: the length of one side of the (2D) image
            n_channels: The number of information channels ("colors") in the image.
            batch_size: The size of batch. Should be specified before training.
        """
        self.sess = sess
        self.batch_size = batch_size
        self.t1_dims = [t1_side_length, t1_side_length, n_channels]

        self.starting_img_dim = t1_side_length / 4 # 4 = 2^(number of conv layers - 1)
        if np.isclose(self.starting_img_dim, int(self.starting_img_dim)):
            self.starting_img_dim = int(self.starting_img_dim)
        else:
            raise Exception("Bad dimension. Haven't figured out how or if to deal with this.")

        self.n_channels = n_channels
        self.imread_mode = 'L' if self.n_channels == 1 else None # not implemented

        self.mni_slice = imageio.get_nifti_image(
            '/home/berleant/dcgan-completion.tensorflow/data/neuro/copy0.nii.gz',
            t1_side_length)

        # batch normalization : deals with poor initialization helps gradient flow
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.visuals_dir = visuals_dir

        self.build_model()
        self.model_name = "T1Generator.model"

    def build_model(self):
        self.t1s = tf.placeholder(
            tf.float32, [None] + self.t1_dims, name='t1_images')
        # Assuming all EPIs in HCP have same spatial dimensions
        self.z = tf.placeholder(tf.float32, [None] + self.EPI_DIMS, name='z')
        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.sampler = self.sampler(self.z)

        self.G_sum = tf.image_summary("G", self.G)

        self.g_loss = tf.reduce_sum(tf.squared_difference(self.G, self.mni_slice))

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)

        self.g_vars = tf.trainable_variables()

        self.saver = tf.train.Saver(max_to_keep=1)

        self.z_seeds = self._seeder()

    def train(self, config):
        counter = tf.Variable(0, name='counter', trainable=False)

        g_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars,
                                                               global_step=counter)

        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.z_sum, self.G_sum, self.g_loss_sum])
        self.writer = tf.train.SummaryWriter(self.log_dir, self.sess.graph)

        start_time = time.time()
        self.load_checkpoint()

        epoch = 0
        n_visuals = 0
        # if n_epochs is not specified, go forever; else do it n_epochs times
        while config.n_epochs is None or epoch < config.n_epochs:
            t1_epi_files = imageio.get_t1_epi_files(config.dataset)

            for t1_file, epi_files in t1_epi_files.items():
                t1_data = imageio.get_nifti_image(t1_file, self.t1_dims[0])
                epi_data = [imageio.get_nifti_image(epi_file, self.EPI_DIMS[0],
                            for epi_file in epi_files]
                epi_data = np.array(epi_data).astype(np.float32)

                # update network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: epi_data})
                self.writer.add_summary(summary_str, epoch)

                errG = self.g_loss.eval({self.z: epi_data}) # ???

                print("Epoch: [%2d] time: %4.4f, g_loss: %.8f"
                      % (epoch, time.time() - start_time, errG))

                if np.power(self.SAVE_VISUALS_EXP, n_visuals) <= epoch:
                    # save visuals in the visuals dir at an ever-decreasing rate
                    n_visuals += 1
                    images = self.sess.run(self.sampler, feed_dict={self.z: self.z_seeds})
                    imageio.save_nifti_images(
                        images, self.VISUALS_DIM, os.path.join(
                            self.visuals_dir, 'train_{:02d}_{:04d}.png'.format(epoch, batch)))
                if np.mod(epoch, self.CHECKPOINT_EPOCHS) == 0:
                    self.save(counter)
            epoch += 1
        print("Training complete.")

    def generator(self, z):
        self.z_, self.h0_w, self.h0_b = ops.linear(
            z, self.starting_img_dim * self.starting_img_dim * self.STARTING_GF_DIM, 'g_h0_lin',
            with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, self.starting_img_dim, self.starting_img_dim, self.STARTING_GF_DIM])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = ops.conv2d_transpose(
            h0,
            [self.batch_size, self.starting_img_dim * 2,
             self.starting_img_dim * 2, self.STARTING_GF_DIM * 2],
            name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = ops.conv2d_transpose(
            h1,
            [self.batch_size, self.starting_img_dim * 4,
             self.starting_img_dim * 4, self.n_channels],
            name='g_h2', with_w=True)

        return tf.nn.tanh(h2)

    def sampler(self, z, y=None):
        ''' cannot be called before generator
        I am not at all confident that removing this would be problematic '''
        tf.get_variable_scope().reuse_variables()

        h0 = tf.reshape(
            ops.linear(z, self.starting_img_dim * self.starting_img_dim * self.STARTING_GF_DIM,
                       'g_h0_lin'),
            [-1, self.starting_img_dim, self.starting_img_dim, self.STARTING_GF_DIM])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = ops.conv2d_transpose(
            h0,
            [self.batch_size, self.starting_img_dim * 2,
             self.starting_img_dim * 2, self.STARTING_GF_DIM * 2],
            name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = ops.conv2d_transpose(
            h1,
            [self.batch_size, self.starting_img_dim * 4,
             self.starting_img_dim * 4, self.n_channels],
            name='g_h2')

        return tf.nn.tanh(h2)

    def save(self, counter):
        self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_dir, self.model_name),
            global_step=counter)

    def load_checkpoint(self):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Found checkpoint!")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint. New model initialized.")

    def _seeder(self):
        return np.random.uniform(-1, 1, # is this why I needed to normalize the T1 data?
                                 size=(self.batch_size, self.INPUT_CONST))
