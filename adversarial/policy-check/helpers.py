import tensorflow as tf
import numpy as np
import json

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for counter in range(int(generated_num / batch_size)):
        if counter % 1000 == 0:
            print "generating fake data..."
            print counter
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for item in generated_samples:
            buffer = ' '.join([str(x) for x in item]) + '\n'
            fout.write(buffer)

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        if it % 100 == 0:
            print "%d / %d finished" %(it, data_loader.num_batch)
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def translator(sentence):
    dict_path = "./datasets/dict.json"
    word_dict = json.load(open(dict_path))
    temp = []
    for item in q:
        for name, idx in word_dict.items():
            if idx == item:
                temp.append(name)
    print temp
