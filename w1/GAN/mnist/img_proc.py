''' preparing the images for the training procedure '''
from __future__ import division

import argparse
import cv2
import os
import scipy
import glob

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(50)

def get_args():
    ''' parsing input arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str)
    parser.add_argument("-type", type=str)
    parser.add_argument("-batch_size", type=int, default=50)
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-img_nums", type=int, default=1)

    args = parser.parse_args()
    return args

def load_img(path):
    ''' resizing the images to 64 by 64 and shuffle axis '''
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (64, 64))) / 127.5 - 1
    # plt.imshow(img)
    # plt.show()
    img = np.rollaxis(img, 2, 0)
    return img

def random_img():
    ''' this will be the image conditioned on by the output of GAN '''
    return np.random.uniform(-1, 1, 100)


def img_demo(path):
    ''' showing the loaded image and the randomly generated image '''
    print "loding image..."
    img_path = glob.glob(os.path.join(path, "*.jpg"))
    Image = np.array(load_img(img_path[0]))
    zmb = random_img()
    return zmb, Image

if __name__ == "__main__":
    args = get_args()
    img_demo(args.path)

    # if args.type == "train":
    #     print args.batch_size
