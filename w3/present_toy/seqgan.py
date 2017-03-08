from gen_dataloader import Gen_Data_loader, Likelihood_Data_loader
from dis_dataloader import Dis_dataloader
from text_classifier import TextCNN
from rollout import ROLLOUT

from pretrain import G_, get_trainable_model, generate_samples
from pretrain import target_loss, pre_train_epoch, pretrain_saver

import model
import numpy as np
import tensorflow as tf
import random
import time
import cPickle

SEQ_LEN = 10
rollout_num = 7
START_TOKEN = 0
SEED = 10

# hyperparams for generator G
EBD_DIM = 1
HID_DIM = 1

PRE_EPC_NUM = 20
TRAIN_ITER = 1
BATCH_SIZE = 32

# Total number of batch
TOTOAL_BATCH = 2#int((200000/10)/BATCH_SIZE)

# hyperparams for D
dis_embedding_dim = 8
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7]#, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100]#, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = .75
dis_l2_reg_lambda = 0.2

# training params
dis_batch_size = 32
dis_num_epochs = 1
dis_alter_epoch = 1

positive_file = 'text_data/text.txt'
_gan_positive_file = 'text_data/partial_real.txt'
negative_file = 'target_generate/eval_file_of_pretrain.txt'
eval_file = 'target_generate/eval_file.txt'
eval_file_of_pretrain = 'target_generate/eval_file_of_pretrain.txt'
save_pre_train_results = "compare/pretrain_results.txt"

generated_num = 300


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    vocab_size = 20525
    best_score = 1000

    gen_data_loader = Gen_Data_loader(batch_size=BATCH_SIZE, sequence_len=SEQ_LEN)
    lik_data_loader = Likelihood_Data_loader(batch_size=BATCH_SIZE, sequence_len=SEQ_LEN)
    dis_dataloader = Dis_dataloader(vocab_size=vocab_size, sequence_len=SEQ_LEN)

    G = get_trainable_model(vocab_size, BATCH_SIZE, EBD_DIM, HID_DIM, SEQ_LEN, START_TOKEN)

    # sequence_len, num_classes, vocab_size, emb_size,
    #  filter_sizes, num_filters, l2_reg_lambda=0.
    with tf.variable_scope("D"):
        cnn = TextCNN(
            sequence_len=SEQ_LEN,
            num_classes=2,
            vocab_size=vocab_size,
            emb_size=dis_embedding_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            l2_reg_lambda=dis_l2_reg_lambda
            )
    cnn_params = [param for param in tf.trainable_variables() if "D" in param.name]

    # training procedure for D
    dis_global_step = tf.Variable(0, name="global_step", trainable=False) #control D
    dis_optimizer = tf.train.AdamOptimizer(1e-3)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

    # configure tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    gen_data_loader.create_batches(positive_file)
    log = open('log/experiment-log.txt', 'w')
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPC_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, G, gen_data_loader)

    # generate sample text from G, after pretrain phase
    to_be_save = generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file_of_pretrain)
    pretrain_saver(save_pre_train_results, to_be_save)
    print "finish pretrain..., check results in compare/"

    # prepare for GAN training, pretrain D
    print '''--------------- training the discriminator ------------'''
    for idx in xrange(dis_alter_epoch):
        print "epoch %d of pre training discriminator... " % idx
        _ = generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
        dis_x_train, dis_y_train = dis_dataloader.load_train_data(_gan_positive_file, negative_file)
        dis_batches = dis_dataloader.batch_iteror(
                zip(dis_x_train, dis_y_train),
                dis_batch_size, dis_num_epochs)

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                feed = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, step = sess.run([dis_train_op, dis_global_step], feed)

            except ValueError:
                pass

    rollout = ROLLOUT(G, 0.3)

    print 'Training GAN: '
    log.write("Reinforcement training for generator... \n")
    for total_batch in xrange(TOTOAL_BATCH):
        print "Reinforcement training for generator... BATCH %d" %total_batch
        for it in xrange(TRAIN_ITER):
            samples = G.generate(sess)
            rewards = rollout.get_reward(sess, samples, rollout_num, cnn)
            feed = {
                G.x: samples, G.rewards: rewards
            }
            _, g_loss = sess.run([G.g_updates, G.g_loss], feed_dict=feed)

        if total_batch % 1 == 0 or total_batch == TOTOAL_BATCH-1:
            _ = generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file_of_pretrain)

        rollout.update_params()

        print "Tarining G"
        for idx_G in xrange(2):
            print "iter %d for training G" % idx_G
            _ = generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
            dis_x_train, dis_y_train = dis_dataloader.load_train_data(_gan_positive_file, negative_file)
            dis_batches = dis_dataloader.batch_iteror(zip(dis_x_train, dis_y_train), dis_batch_size, 3)

            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, step = sess.run([dis_train_op, dis_global_step], feed)

                except ValueError:
                    pass
    log.close()

if __name__ == "__main__":
    main()
