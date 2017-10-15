from trim_generator import Generator as G
from discriminator import Discriminator as D
from dataloader import g_data_loader as G_ld
from dataloader import full_loader as F_ld
from dataloader import trimed_g_loader as t_g_loader
from rollout import ROLLOUT

import numpy as np
import tensorflow as tf
import random
import os

import helpers as hp
import fuse.trim_fuser as fuser


EMB_DIM = 256
HID_DIM = 64
START_TOKEN = 0
PRE_EPOCH_NUM = 80
SEED = 1234
BATCH_SIZE = 1
VOCAB_SIZE = 20524
LR = 0.01

dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3]
dis_num_filters = [10, 50, 100]
dis_dropout_keep_prob = 0.85
dis_l2_reg_lambda = 0.05
dis_batch_size = 64

out_file = "./save/generated.txt"
pretrain_save = "/pretrained/"+str(PRE_EPOCH_NUM)+"model.ckpt"
pretrain_loss_save = open("./pretrained_log/pretrain_loss_save.txt", "w")
pretrain_loss = []

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    full_loader = F_ld()
    Q, real_A, candidates, cand_max_len, QA_IDX = full_loader.pad_candidates()

    generator = G(cand_max_len=cand_max_len, emb_dim=EMB_DIM)
    discriminator = D(len(real_A), dis_filter_sizes, dis_num_filters)
    pre_G_dataloader = t_g_loader(BATCH_SIZE, cand_max_len, candidates)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()

    if len(os.listdir("./pretrained/")) <= 2:
        print "Pretraining the generator ... "
        for epoch in xrange(PRE_EPOCH_NUM):
            loss = hp.pre_train_epoch(sess, generator, pre_G_dataloader)
            if epoch % 1 == 0:
                cwd = os.getcwd()
                print "This is generator pretrain epoch %d: " % epoch
                print loss
                pretrain_loss.append(loss)

        for item in pretrain_loss:
            pretrain_loss_save.write(str(item)+", ")
        pretrain_loss_save.close()

        print "====== Finish pretraining of generator ======"

        print "Pretraining the discriminator ... "
        nega_A = hp.generate_samples(sess, generator, candidates, out_file)[0]
        feed = hp.get_dict_D(discriminator, nega_A, real_A)

        cwd = os.getcwd()
        # saver.save(sess, cwd+pretrain_save)
        # print "Pretraining G model saved!"
    else:
        cwd = os.getcwd()
        # saver.restore(sess, cwd+pretrain_save)
        # print "Pretrained G model restored!"

    for _ in xrange(70):
        _, pre_d_loss = sess.run([discriminator.train_op, discriminator.loss], feed_dict=feed)
        print pre_d_loss
    print "====== Finish pretraining of discriminator ======"

    adversarial_g_loss_records = "./adversarial_log/"+str(QA_IDX)+"_"+str(PRE_EPOCH_NUM)+"_generator_fuse.txt"
    adversarial_d_loss_records = "./adversarial_log/"+str(QA_IDX)+"_"+str(PRE_EPOCH_NUM)+"_discriminator_fuse.txt"
    adversarial_reward_records = "./adversarial_log/"+str(QA_IDX)+"_"+str(PRE_EPOCH_NUM)+"_rewards.txt"

    rewards_records = []
    adversarial_g_loss = []
    adversarial_d_loss = []
    print "///// begin of adversarial training /////"
    rollout = ROLLOUT(generator, 0.8)
    for _ in xrange(30):
        for g_it in xrange(3):
            samples = generator.generate(sess, candidates)
            rewards = rollout.get_reward(sess, samples, cand_max_len-1, discriminator, candidates)
            feed = { generator.x: samples,
                     generator.rewards: rewards,
                     generator.begin_ad: True,
                     rollout.lstm.fuser.input_ph: np.expand_dims(np.asarray(candidates), axis=0)}
            _ = sess.run(generator.g_updates, feed_dict=feed)

            ''' begin teacher forcing '''
            rewards_records.append(np.mean(rewards))
            rewards_ = np.ones(rewards.shape)
            feed_ = { generator.x: real_A,
                      generator.rewards: rewards_,
                      generator.begin_ad: True,
                      rollout.lstm.fuser.input_ph: np.expand_dims(np.asarray(candidates), axis=0)}
            _ = sess.run(generator.g_updates, feed_dict=feed)
        print rewards
        rollout.update_params()

        for d_it in xrange(1):
            nega_A = hp.generate_samples(sess, generator, candidates, out_file)[0]
            feed = hp.get_dict_D(discriminator, nega_A, real_A)
            _, pre_d_loss, d_pred, d_real, d_losses = sess.run([discriminator.train_op, discriminator.loss, discriminator.predictions, discriminator.input_y, discriminator.losses], feed_dict=feed)

        print "GAN d loss"
        print pre_d_loss
        adversarial_d_loss.append(pre_d_loss)

        print "------ round summary --------"
        g_gan_loss = hp.loss_checker(sess, generator, pre_G_dataloader, candidates)
        adversarial_g_loss.append(g_gan_loss)
        print "Results of GAN:"
        print g_gan_loss
        samples = generator.generate(sess, candidates)
        hp.translator(Q, real_A, samples[0])
        print "-----------------------------"

    ''' write out the results '''
    with open(adversarial_g_loss_records, "w") as f:
        for item in adversarial_g_loss:
            f.write(str(item)+", ")
    f.close()
    with open(adversarial_d_loss_records, "w") as f:
        for item in adversarial_d_loss:
            f.write(str(item)+", ")
    f.close()
    with open(adversarial_reward_records, "w") as f:
        for item in rewards_records:
            f.write(str(item)+", ")
    f.close()

if __name__ == "__main__":
    main()
