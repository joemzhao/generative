from __future__ import division
from overrides import overrides
from scipy.stats import norm
from numeric import mlp, momentum

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mu = -1.
sigma = 1.

TRAIN_ITERS = 10000
batch_size = 100

with tf.variable_scope("D_pre"):
    input_node = tf.placeholder(tf.float32, shape=(batch_size, 1))
    train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))
    D, params_D_pre = mlp(input_node, 1)
    loss = tf.reduce_mean(tf.square(D - train_labels))

optimizer = momentum(loss, None)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def plot_d0(D, input_node):
    f, ax = plt.subplots(1)
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label="p_data")
    r = len(xs)
    print r
    xs = np.linspace(-5, 5, r)
    ds = np.zeros((r, 1))
    for i in range(int(r/batch_size)):
        x = np.reshape(xs[batch_size*i : batch_size*(i+1)], (batch_size, 1))
        ds[batch_size*i:batch_size*(i+1)] = sess.run(D, {input_node: x})
    ax.plot(xs, ds, label="Decision Boundary")
    ax.set_ylim(0, 1.1)
    plt.legend()
    plt.title('Initial Decision Boundary')

plot_d0(D, input_node)
plt.show()

lh = np.zeros(1000)
for i in xrange(1000):
    # generate loss, stratch to normal to make GAN performed better
    d = (np.random.random(batch_size) - 0.5) * 10.0
    labels = norm.pdf(d, loc=mu, scale=sigma) # actually value -- true data

    lh[i], _ = sess.run([loss, optimizer],
        {input_node: np.reshape(d,(batch_size,1)),
        train_labels: np.reshape(labels, (batch_size, 1))})

plt.plot(lh)
plt.title("Training Loss")

plot_d0(D,input_node)

weightsD=sess.run(params_D_pre)
sess.close()

with tf.variable_scope("G"):
    z_node = tf.placeholder(tf.float32, shape=(batch_size, 1))
    G, params_G = mlp(z_node, 1)
    G = tf.multiply(5., G)

with tf.variable_scope("D") as scope:
    x_node = tf.placeholder(tf.float32, shape=(batch_size, 1))
    fc, params_D = mlp(x_node, 1)
    D1 = tf.maximum(tf.minimum(fc, .99), .01)
    scope.reuse_variables()
    fc, params_D = mlp(G, 1)
    D2 = tf.maximum(tf.minimum(fc, .99), .01)

obj_d = tf.reduce_mean(tf.log(D1) + tf.log(1-D2))
obj_g = tf.reduce_mean(tf.log(D2))

opt_d = momentum(1-obj_d, params_D)
opt_g = momentum(1-obj_g, params_G)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# copy the paramesters from pretrained D
for idx, v in enumerate(params_D):
    sess.run(v.assign(weightsD[idx]))

M = batch_size
def plot_fig():
    # plots pg, pdata, decision boundary
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

    # decision boundary
    r=5000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in same minibatch
    for i in xrange(int(r/M)):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D1,{x_node: x})

    ax.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs=np.linspace(-5,5,r)
    gs=np.zeros((r,1)) # generator function
    for i in xrange(int(r/M)):
        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
        gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')

    # ylim, legend
    ax.set_ylim(0,1.1)
    plt.legend()

plot_fig()
plt.title('Before Training')
plt.show()

k=2
histd, histg= np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)

for i in xrange(TRAIN_ITERS):
    for j in xrange(k):
        x = np.random.normal(mu, sigma, M)
        x.sort()
        z = np.linspace(-5.0, 5.0, M)+np.random.random(M)*0.01
        histd[i],_ = sess.run([obj_d, opt_d], {x_node: np.reshape(x, (M, 1)),
                                                z_node: np.reshape(z, (M, 1))})
    z = np.linspace(-5.0, 5.0, M) + np.random.random(M) * 0.01
    histg[i], _ = sess.run([obj_g,opt_g], {z_node: np.reshape(z,(M, 1))})

    if i % (TRAIN_ITERS//10) == 0:
        print(float(i)/float(TRAIN_ITERS))

plt.plot(range(TRAIN_ITERS),histd, label='obj_d')
plt.plot(range(TRAIN_ITERS), 1-histg, label='obj_g')
plt.legend()
plot_fig()
plt.show()
