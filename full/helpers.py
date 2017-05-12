import tensorflow as tf
import numpy as np
import json

def generate_samples(sess, trainable, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in xrange(int(generated_num / batch_size)):
        generated_samples.extend(trainable.generate(sess))
    with open(output_file, "w") as fout:
        for item in generated_samples:
            temp = ' '.join([str(x) for x in item]) + '\n'
            fout.write(temp)
    fout.close()
    return generated_samples

def pre_train_epoch(sess, trainable, data_loader, candidates):
    '''
    Since this is for pretraining, for placeholder in the fusing operator (input candidates) we will just use the first set of candidates.
    '''
    supervised_g_loss = []
    data_loader.reset_pointer()

    for batch in xrange(data_loader.num_batch):
        if batch % 5 == 0 and batch > 0:
            print "%d / %d" % (batch, data_loader.num_batch)
            print "Training loss : ", np.mean(supervised_g_loss)

        next_bc = data_loader.next_batch()

        ''' [pretrain_update, pretrain_loss] '''
        _, g_loss = trainable.pretrain_step(sess, next_bc, candidates)

        supervised_g_loss.append(g_loss)

    return np.mean(supervised_g_loss)

def translator(q, a, a_):
    dict_path = "./datasets/dict.json"
    word_dict = json.load(open(dict_path))
    temp = []
    print "Question:"
    for item in q:
        for name, idx in word_dict.items():
            if idx == item:
                temp.append(name)
    print temp
    temp = []
    print "Real answer:"
    for item in a:
        for name, idx in word_dict.items():
            if idx == item:
                temp.append(name)
    print temp
    temp = []
    print "From GAN:"
    for item in a_:
        for name, idx in word_dict.items():
            if idx == item:
                temp.append(name)
    print temp
