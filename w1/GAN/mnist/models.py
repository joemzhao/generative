from __future__ import division

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras import initializations

import numpy as np

def initNormal(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name)

def build_G(Optimizer, noise_dim):
    print "Building the generator..."
    G = Sequential()
    G.add(Dense(256, input_dim=noise_dim, init=initNormal))
    G.add(LeakyReLU(0.2))
    G.add(Dense(512))
    G.add(LeakyReLU(0.2))
    G.add(Dense(1024))
    G.add(LeakyReLU(0.2))
    G.add(Dense(784, activation="tanh")) # 28 by 28
    G.compile(loss="binary_crossentropy", optimizer=Optimizer)

    return G

def build_D(Optimizer, noise_dim):
    print "Building the discriminator..."
    D = Sequential()
    D.add(Dense(1024, input_dim=784, init=initNormal))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))
    D.add(Dense(512))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))
    D.add(Dense(256))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))
    D.add(Dense(1, activation="sigmoid"))
    D.compile(loss="binary_crossentropy", optimizer=Optimizer)

    return D
