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
        # print l[i:i+n]

def train_model(path, batch_size, epochs):
    np.random.seed(50)

    Noisy, Image = preproc.img_demo(path)
    print "Finish obtaining images..."
    
    Batches = [b for b in get_data_batch(Image, batch_size)]
    print "Finish getting batches..."

    G = models.G()
    D = models.D()
    G_D = models.G_D(G, D)
    print "Finish connecting nns..."

    adam_G = Adam(lr=0.02, beta_1=0.05, beta_2=0.05, epsilon=0.001)
    adam_D = Adam(lr=0.02, beta_1=0.05, beta_2=0.05, epsilon=0.001)

    G.compile(loss="binary_crossentropy", optimizer=adam_G)
    G_D.compile(loss="binary_crossentropy", optimizer=adam_G)

    D.trainable = True
    D.compile(loss="binary_crossentropy", optimizer=adam_D)

    print "Start training..."
    print "number of batches is: ", len(Batches[0])
    print "Batch size is: ", batch_size

    G_D_margin = 0.10

    for epoch in xrange(epochs):
        print "Epoch: ", epoch

        if epoch == 0:
            if os.path.exists("G_weights") and os.path.exists("D_weights"):
                print "Loading saved weights..."
                G.load_weights("G_weights")
                D.load_weights("D_weights")
            else:
                print "No existing weigths availabel!"

        for index, img_batch in enumerate(Batches[0]):
            print "Epoch: ", epoch, "Batch: ", index
            noise_batch = np.array([ Noisy for n in xrange(len(img_batch))])
            # print noise_batch






if __name__ == "__main__":
    args = preproc.get_args()
    train_model(args.path, 5, 1)
