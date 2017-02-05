''' tensorflow-ish operations '''

import tensorflow as tf

# create scratch variable scope to get around reuse=True issues with
# temporary variables
# (hack around https://github.com/tensorflow/tensorflow/issues/5827)
with tf.variable_scope("scratch", reuse=False) as scratch_varscope:
    SCRATCH_VARSCOPE = scratch_varscope
assert SCRATCH_VARSCOPE == scratch_varscope
assert SCRATCH_VARSCOPE.reuse == False

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                            initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                             initializer=tf.random_normal_initializer(1., 0.02))
                with tf.variable_scope(SCRATCH_VARSCOPE) as scratch_varscope:
                    assert scratch_varscope.reuse == False
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                    self.ema_mean = self.ema.average(batch_mean)
                    self.ema_var = self.ema.average(batch_var)
                    with tf.control_dependencies([ema_apply_op]):
                        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def conv2d_transpose(input_, output_shape,
                     k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable(
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
