from generator import Generator as G
from discriminator import Discriminator as D
from dataloader import g_data_loader as G_ld
from rollout import ROLLOUT

import numpy as np
import tensorflow as tf
import random
import helpers

import fuse.fuse_utils as fu
import fuse.fusing as fusing
import fuse.fuser as fuser


EMB_DIM = 256
HID_DIM = 64
SEQ_LEN = 20
START_TOKEN = 0
PRE_EPOCH_NUM = 1
SEED = 1234
BATCH_SIZE = 1
VOCAB_SIZE = 20525
LR = 0.01

dis_embedding_dim = 64
dis_filter_sizes = [1, 2]
dis_num_filters = [100, 200]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

data_path = "./datasets/bbt_concate.txt"
out_file = "./save/generated.txt"

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    FUSER = fuser.Fuser() # QA batches are included
    generator = G(FUSER, VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HID_DIM, SEQ_LEN, START_TOKEN, LR)
    discriminator = D(sequence_length=generator.fuser.candidate_max_length, num_classes=2, vocab_size=VOCAB_SIZE, embedding_size=dis_embedding_dim, filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    G_dataloader = G_ld(BATCH_SIZE, generator.fuser.candidate_max_length, data_path)
    G_dataloader.create_batches()

    for epoch in xrange(PRE_EPOCH_NUM):
        loss = helpers.pre_train_epoch(sess, generator, G_dataloader)
        if epoch % 5 == 0:
            print "This is generator pretrain epoch %d: " % epoch
            print loss

    nega_A = helpers.generate_samples(sess, generator, 1, 1, out_file)[0]
    real_A = np.array(FUSER.A_batches[0][0][:6])

    posilabel = [[1, 0]]
    negalabel = [[0, 1]]
    feed = {discriminator.input_x: np.array([nega_A, real_A]),
            discriminator.input_y: np.concatenate([posilabel, negalabel], 0),
            discriminator.dropout_keep_prob: 0.9}

    for epoch in xrange(50):
        _, _ = sess.run([discriminator.train_op, discriminator.loss], feed_dict=feed)

    print "Finish pretrain D."
    print "==== begin of adversarial training ===="
    print "Target is [ 72  21  40 259  14  15 ] "
    rollout = ROLLOUT(generator, 0.8)
    MAX_CAN = generator.fuser.candidate_max_length
    ''' Adversarial Training '''
    for _ in xrange(5):
        for it in xrange(3):
            samples = generator.generate(sess)
            candidates_tofeed = rollout.lstm.fuser.get_candidates_tofeed()
            rewards = rollout.get_reward(sess, samples, MAX_CAN-1, discriminator)
            feed = {    generator.x: samples,
                        generator.rewards: rewards,
                        rollout.lstm.fuser.input_ph: candidates_tofeed}
            _ = sess.run(generator.g_updates, feed_dict=feed)
            samples = generator.generate(sess)
            if it % 10  == 0:
                loss = helpers.pre_train_epoch(sess, generator, G_dataloader)
                print "G adversarial loss", loss
                print samples

        rollout.update_params()

        for _ in xrange(1):
            nega_A = helpers.generate_samples(sess, generator, 1, 1, out_file)[0]
            real_A = np.array(FUSER.A_batches[0][0][:6])
            feed = {discriminator.input_x: np.array([nega_A, real_A]),
                    discriminator.input_y: np.concatenate([posilabel, negalabel], 0),
                    discriminator.dropout_keep_prob: 0.9}
            _ = sess.run(discriminator.train_op, feed)

        print "------ round summary --------"
        g_gan_loss = helpers.pre_train_epoch(sess, generator, G_dataloader)
        print "Results of GAN:"
        print g_gan_loss
        print "-----------------------------"

if __name__ == "__main__":
    main()
