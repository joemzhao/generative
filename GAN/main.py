from __future__ import division
from keras.optimizers import Adam, SGD, RMSprop

import cv2
import numpy as np
import matplotlib.pyplot as plt
import img_proc as preproc

import glob
import os

import models

def get_data_batch(l, n):
    ''' yield n-sized batch from l '''
    for i in range(0, len(l), n):
        yield l[i:i+n]

def train_model(path, batch_size, epochs):
    np.random.seed(50)

    print "Obtaining images..."
    Noisy, Image = preproc.img_demo(path)

    print "Getting batches..."
    Batches = [b for b in get_data_batch(Image, batch_size)]

    print "Connecting nns..."
    G = models.G()
    D = models.D()
    G_D = models.G_D(G, D)

if __name__ == "__main__":
    args = preproc.get_args()
    train_model(args.path, 10, 1)
