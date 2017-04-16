import tensorflow as tf
import numpy as np
import os
import sys

import data_helper as data_helper

from D import discriminator as D
from configers import config_D

def create_model(sess, config, is_training):
    model = D(config=config, is_training=True)
    sess.run(tf.global_variables_initializer())
    return model

def evaluate(model, sess, data, batch_size, global_steps=None):
    correct_num = 0
    total_num = len(data[0])

    for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data, batch_size=batch_size)):
        fetches = model.correct_num
        feed_dict = {
            model.input_data: x,
            model.target: y,
            model.mask_x: mask_x
        }
        state = sess.run(model.initial_state)
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        count = sess.run(fetches, feed_dict)
        correct_num += count

    accuracy = float(correct_num)/total_num
    return accuracy

def run_epoch(model, sess, data, global_steps,
              valid_model, valid_data, batch_size):
    for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data, batch_size=batch_size)):
        feed_dict = {
            model.input_data: x,
            model.target: y,
            model.mask_x: mask_x
        }
        fetches = [model.cost, model.accuracy, model.train_op]
        state = sess.run(model.initial_state)
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, accuracy, _ = sess.run(fetches, feed_dict)
        valid_accuracy = evaluate(valid_model, sess, valid_data, batch_size, global_steps)
        if global_steps%10==0:
            print("step %i, train cost: %f, train accuracy:%f, valid accuracy is %f " % (global_steps, cost, accuracy, valid_accuracy))

        global_steps += 1

    return global_steps

def train_step(config_disc, config_evl):
    config = config_disc
    eval_config = config_evl
    eval_config.keep_prob = 1.

    train_data, valid_data, test_data = data_helper.load_data(True, config.max_len, batch_size=config.batch_size)

    print "Start training..."
    with tf.Graph().as_default(), tf.Session() as sess:
        print "Training model..."
        initializer = tf.random_uniform_initializer(-1*config.init_scale, 1*config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = create_model(sess, config, is_training=True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = create_model(sess, eval_config, is_training=False)
            test_model = create_model(sess, eval_config, is_training=False)

        tf.global_variables_initializer().run()
        global_steps = 1

        for step in xrange(config.num_epoch):
            print "Epoch: %d" % step
            global_steps = run_epoch(model, sess, train_data, global_steps,
                            valid_model, valid_data, config_disc.batch_size)
            test_accuracy=evaluate(test_model, sess, test_data, config_disc.batch_size)
            print("the test data accuracy is %f"%test_accuracy)

if __name__ == "__main__":
    D_configure = config_D()
    train_step(D_configure, D_configure)
