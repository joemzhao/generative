import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def write_results(path, dloss, gloss):
    with open(path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows([dloss])
        writer.writerows([gloss])
    f.close()

def plot_loss(path, epochs, dloss, gloss):
    plt.figure(figsize=(10, 10))
    plt.plot(dloss, label="D loss")
    plt.plot(gloss, label="G loss")
    plt.xlabel("Epoch index")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path+"_epoch_%d"%epochs)

def get_image_G(G, epoch, path, img_nums=36, dim=(6, 6), figsize=(10, 10), noise_dim=10):
    noise = np.random.normal(0, 1, size=[img_nums, noise_dim])
    generated_img = G.predict(noise)
    generated_img = generated_img.reshape(img_nums, 28, 28)

    plt.figure(figsize=figsize)
    for i in xrange(generated_img.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_img[i], interpolation="nearest", cmap="gray_r")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path+"gan_generated_img_e=%d.png"%epoch)
