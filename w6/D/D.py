import tensorflow as tf
import numpy as np
import os
import sys

# from utils.nn_config import D_config
# from loaders.reader import data_reader


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
        self.target = tf.placeholder(tf.int64, [None])
        self.mask_x = tf.placeholder(tf.float32, [self.max_len, None])
        self.init_layers()

    def init_layers(self):
        '''
        Define layers in the network. Embedding + RNNcells + Output layer
        '''

        '''Embdedding Layer'''
        with tf.device("/cpu:0"), tf.name_scope("embed_layer"):
            self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.emb_dim], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.Embedding, self.input_data)

        '''Define LSTM cells and make layers '''
        single_cell = tf.contrib.rnn.LSTMCell(
            num_units=self.hidden_dim,
            forget_bias=0.,
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
            outputs = tf.reduce_sum(outputs, 0)/(tf.reduce_sum(self.mask_x, 0)[:, None])

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
            logits=self.logits,
            labels=self.target
        )
        self.cost = tf.reduce_mean(self.loss)

        self.prediction = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(self.prediction, self.target)
        self.correct_nums = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        self.step_()

    def step_(self):
        '''
        Define optimizers and run the model
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = optimizer.compute_gradients(self.cost)
        clip_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(clip_gradients)

if __name__ == "__main__":
    lstm = LSTM_(emb_dim=10, hidden_dim=100,
                       vocab_size=2000, batch_size=128, lr=0.01, num_layers=1, keep_prob=.95, max_len=20)
