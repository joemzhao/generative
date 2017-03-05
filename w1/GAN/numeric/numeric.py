from __future__ import division
from overrides import overrides
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# parameters for the target distribution
mu = -1.
sigma = 1.
xs = np.linspace(-5., 5., 10000)
plt.plot(xs, norm.pdf(xs, loc=mu, scale=sigma))
# plt.show()

TRAIN_EPCS = 10000
batch_size = 100

# NN for D and G
def mlp(input, output_dim):
    w1 = tf.get_variable("w0", [input.get_shape()[1], 20], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b0", [20], initializer=tf.constant_initializer(0.))

    w2 = tf.get_variable("w1", [20, 10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b1", [10], initializer=tf.constant_initializer(0.))

    w3 = tf.get_variable("w2", [10, output_dim], initializer=tf.random_normal_initializer())
    b3 = tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.))

    fc1 = tf.nn.tanh(tf.matmul(input, w1) + b1)
    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
    fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)

    return fc3, [w1, b1, w2, b2, w3, b3]

# def an optimizer
def momentum(loss, var_list):
    batch = tf.Variable(0)
    lr = tf.train.exponential_decay(
        0.001,
        batch,
        TRAIN_EPCS,
        .95,
        staircase=True)
    optimizer = tf.train.MomentumOptimizer(lr, 0.1).minimize(loss, global_step=batch, var_list=var_list)
    return optimizer
