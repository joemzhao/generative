import os
import csv
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
