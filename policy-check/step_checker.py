from trim_generator import Generator as G
from discriminator import Discriminator as D
from dataloader import Gen_Data_loader as G_ld
from dataloader import Dis_dataloader as D_ld
from rollout import ROLLOUT

import numpy as np
import tensorflow as tf
import random
import os

import helpers as hp


EMB_DIM = 64
HID_DIM = 64
START_TOKEN = 0
PRE_EPOCH_NUM = 0
SEED = 1234
BATCH_SIZE = 2
SEQ_LENGTH = 20
VOCAB_SIZE = 20525
LR = 0.01

dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5]
dis_num_filters = [100, 200, 200, 200, 200]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

real_data_path = "./datasets/bbt_concate_full.txt"
nega_data_path = "./save/generated.txt"

pretrain_save = "/pretrained/"+str(PRE_EPOCH_NUM)

pretrain_loss = []
pretrain_loss_d = []
g_pretrain_loss_save = open("./log/g_pretrain_loss_save.txt", "w")
d_pretrain_loss_save = open("./log/d_pretrain_loss_save.txt", "w")

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    generator = G(batch_size=BATCH_SIZE)
    discriminator = D(filter_sizes=dis_filter_sizes, num_filters=dis_num_filters)

    G_dataloader = G_ld(BATCH_SIZE)
    D_dataloader = D_ld(BATCH_SIZE)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    totoal_words = G_dataloader.create_batches(real_data_path, SEQ_LENGTH)
    totoal_words = BATCH_SIZE * 200

    print "Pretraining the generator ... "
    for epoch in xrange(PRE_EPOCH_NUM):
        loss = hp.pre_train_epoch(sess, generator, G_dataloader)
        if epoch % 5 == 0:
            print "This is generator pretrain epoch %d: " % epoch
            print loss
            pretrain_loss.append(loss)
    for item in pretrain_loss:
        g_pretrain_loss_save.write(str(item)+",")
    g_pretrain_loss_save.close()
    print "====== Finish pretraining of generator ======"

    print "Pretraining the discriminator ... "
    # print totoal_words
    hp.generate_samples(sess, generator, BATCH_SIZE, totoal_words, nega_data_path)
    d_batch = D_dataloader.load_train_data(real_data_path, nega_data_path, SEQ_LENGTH)

    d_batch_losses = []
    for epoch in range(2):
        D_dataloader.reset_pointer()
        for batch in xrange(D_dataloader.num_batch):
            x_batch, y_batch = D_dataloader.next_batch()
            feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
            _, d_pre_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
            d_batch_losses.append(d_pre_loss)
            if batch % 200 == 0:
                print "discriminator pretrain batch %d/%d: " % (batch, d_batch)
        pretrain_loss_d.append(np.mean(d_batch_losses))
        d_batch_losses = []

    for item in pretrain_loss_d:
        d_pretrain_loss_save.write(str(item)+",")
    d_pretrain_loss_save.close()

    cwd = os.getcwd()
    saver.save(sess, cwd+pretrain_save+"_%d_model.ckpt"%epoch)
    print "Pretraining model saved!"

    print "///// begin of adversarial training /////"
    adversarial_g_loss = []
    adversarial_d_loss = []
    rollout = ROLLOUT(generator, 0.8)
    for iteration in xrange(50):
        if iteration % 5 == 0:
            print "This is epoch %d of adversarial training" % iteration

        for g_it in xrange(5):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, ad_g_loss = sess.run([generator.g_updates, generator.pretrain_loss], feed_dict=feed)

        adversarial_g_loss.append(ad_g_loss)
        rollout.update_params()


        hp.generate_samples(sess, generator, BATCH_SIZE, totoal_words, nega_data_path)
        d_batch = D_dataloader.load_train_data(real_data_path, nega_data_path, SEQ_LENGTH)
        for _ in range(1):
            D_dataloader.reset_pointer()
            for it in xrange(D_dataloader.num_batch):
                x_batch, y_batch = D_dataloader.next_batch()
                feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                _, d_pre_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
        adversarial_d_loss.append(d_pre_loss)
        print "-----  GAN summary -----"
        print "avg g loss %f "%np.mean(adversarial_g_loss)
        print "avg d loss %f "%np.mean(adversarial_d_loss)

    with open("./log/ad_g_loss.txt", "w") as f:
        for item in adversarial_g_loss:
            f.write(str(item)+",")
    f.close()

    with open("./log/ad_d_loss.txt", "w") as f:
        for item in adversarial_d_loss:
            f.write(str(item)+",")
    f.close()

if __name__ == "__main__":
    main()
