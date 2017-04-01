import tensorflow as tf
import numpy as np
import os
import sys

import loaders.reader
import utils.nn_config.D_config

class LSTM_(object):
    '''
    An LSTM used as the discriminator within the GAN model.
    '''
    def __init__(self, emb_dim, hidden_dim,
                       vocab_size, batch_size, lr, num_layers, keep_prob, max_len):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.max_len = max_len
        self.class_num = 2
        self.init_placeholders()

    def init_placeholders(self):
        self.input_data = tf.placeholder(tf.int32, [None, self.max_len])
        self.target = tf.placeholder(tf.int32, [None])
        self.mask_x = tf.placeholder(tf.float32, [self.max_len, None])
        self.init_layers()

    def init_layers(self):
        '''
        Define layers in the network. Embedding + RNNcells + Output layer
        '''

        '''Embdedding Layer'''
        with tf.device("/cpu:0"), tf.name_scope("embed_layer"):
            self.Embedding =
                tf.get_variable("Embedding", [self.vocab_size, self.emb_dim], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.Embedding, self.input_data)

        '''Define LSTM cells and make layers '''
        single_cell = tf.contrib.rnn.LSTMCell(
            num_units=self.hidden_dim
            forget_bias=0.
            state_is_tuple=True
        )
        if self.keep_prob<1:
            single_cell = tf.contrib.rnn.DropoutWrapper(
                cell=single_cell,
                output_keep_prob=self.keep_prob
            )
        self.cells = tf.contrib.rnn.MultiRNNCell(
            cells=[single_cell]*self.num_layers,
            state_is_tuple=True
        )
        self.init_state = self.cells.zero_state(self.batch_size, dtype=tf.float32)

        state = self.init_state
        outputs = []
        with tf.variable_scope("lstm_layer"):
            for idx in xrange(self.max_len):
                if idx > 0:
                    tf.get_variable_scope().reuse_variables()
                (lstm_output, state) = self.cells(inputs[:, idx, :], state)
                outputs.append(lstm_output)
        outputs = outputs * self.mask_x[:, :, None]

        ''' mean pooling layer '''
        with tf.name_scope("mean_pooling_layer"):
            outputs = tf.reduce_sum(output, 0)/(tf.reduce_sum(self.mask_x, 0)[:, None])

        ''' soft max layer and outputs results '''
        with tf.name_scope("softmax_and_output_layer"):
            softmax_w = tf.get_variable(
                "softmax_w",
                [self.hidden_dim, self.class_num],
                dtype=tf.float32
            )
            softmax_b = tf.get_variable(
                "softmax_b",
                [self.class_num],
                dtype=tf.float32
            )
            self.logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.make_predict()

    def make_predict(self):
        '''
        Make predictions. Calculate loss and accuracy
        '''
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.logits,
            self.target
        )
        self.cost = tf.reduce_mean(self.loss)

        self.prediction = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(self.prediction, self.target)
        self.correct_nums = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        self.loss_summary = tf.summary.scalar("loss", self.cost)
        self.accu_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.step_()

    def step_(self):
        '''
        Define optimizers and run the model
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = optimizer.compute_gradients(self.cost)
        clip_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(clip_gradients)

        '''
        Tracking the gradients and loss
        '''
        tvars = tf.trainable_variables()
        grad_summaries = []
        for g, v in zip(gradients, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        self.summary = tf.summary.merge([
            self.loss_summary,
            self.accuracy_summary,
            self.grad_summaries_merged
        ])
