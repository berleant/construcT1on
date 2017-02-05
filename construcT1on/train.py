#!/usr/bin/env python2
""" Dream up T1 nifti images. Heavily modified from
https://github.com/bamos/dcgan-completion.tensorflow """

import os

import tensorflow as tf

from construcT1on.generator import T1Generator

def main():
    flags = tf.app.FLAGS
    flags.DEFINE_integer("n_epochs", None, "How many epochs to train")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
    flags.DEFINE_integer("batch_size", 64, "The size of batch images")
    flags.DEFINE_integer("image_size", 128, "The size of image to use")
    flags.DEFINE_string("dataset", "/scratch/PI/russpold/data/HCP", "Dataset directory.")
    flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
    flags.DEFINE_string("visuals_dir", "visuals", "Directory name to save the visuals")
    flags.DEFINE_string("anatomical_template", "Disk1of5/*/MNINonLinear/T1w_restore.2.nii.gz",
                        "Template for anatomical files, to be used by python's `glob.glob`")
    FLAGS = flags.FLAGS

    for directory in [FLAGS.checkpoint_dir, FLAGS.visuals_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        t1generator = T1Generator(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)
        last_step = t1generator.train(FLAGS)
        t1generator.save(last_step)

main()
