from __future__ import division
from tqdm import tqdm
from time import gmtime, strftime

from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

import models
import helpers

os.environ["KERAS_BACKEND"]="tensorflow"
K.set_image_dim_ordering('th')

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=128)
    parser.add_argument("-e", type=int, default=20)
    return parser.parse_args()

out_path = os.getcwd() + "/results/"
img_path = os.getcwd() + "/imgs/"
mod_path = os.getcwd() + "/saved_model/"
now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# for replicate
np.random.seed(1000)

# load data
# 60000 training images, each corresponds to a 28-by-28 dimentions vector
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

# def an optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# get G and D
noise_dim = 10

G = models.build_G(adam, noise_dim)
D = models.build_D(adam, noise_dim)

# connect G and D to get GAN
# weights in D must be frozen when updating the weights in G
D.trainable = False
ganInput = Input(shape=(noise_dim,))
x = G(ganInput)
ganOutput = D(x)
gan = Model(input=ganInput, output=ganOutput) # passing tensor to get gan
gan.compile(loss="binary_crossentropy", optimizer=adam)
print "finish building GAN..."

def save_model(epoch_index):
    G.save(mod_path+"gan_G_epoch_%d_.h5" % epoch_index)
    G.save_weights(mod_path+"gan_G_epoch_%d_.h5" % epoch_index)

    D.save(mod_path+"gan_D_epoch_%d_.h5" % epoch_index)
    D.save_weights(mod_path+"gan_D_epoch_%d_.h5" % epoch_index)

dLosses = []
gLosses = []

def train(epochs=2, batch_size=128):
    batch_count = int(X_train.shape[0]/batch_size)
    print "Start training ..."
    print "Total number of epochs: ", epochs
    print "Batch size: ", batch_size
    print "Number of batches: ", batch_count

    for e in xrange(1, epochs+1):
        # go through epochs
        print "-"*15, "Epoch %d" %e, "-"*15

        for _ in tqdm(xrange(batch_count)):
            # generate noise then get predict images from G
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
            image_batch = X_train[np.random.randint(0, X_train.shape[0],
                                           size=batch_size)]
            Ged_images = G.predict(noise)

            # concatenate the real and generated images, to feed to D
            X = np.concatenate([image_batch, Ged_images])
            yDis = np.zeros(2*batch_size)
            yDis[:batch_size] = .9

            # train D firstly
            D.trainable = True
            dloss = D.train_on_batch(X, yDis)

            # train G, freeze weights in D firstly
            D.trainable = False
            noise = np.random.normal(0, 1, size=[batch_size, noise_dim])
            yG = np.ones(batch_size)
            gloss = gan.train_on_batch(noise, yG)

        # for each batch, store loss of G, D
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e%10 == 0 or e == 1:
            save_model(e)

    print dLosses
    print gLosses

    helpers.write_results(out_path+now+"bs=%d"%batch_size+"e=%d"%epochs+".csv",
                                    dLosses, gLosses)

if __name__ == "__main__":
    args = Parser()
    train(epochs=args.e, batch_size=args.bs)
