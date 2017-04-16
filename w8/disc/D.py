from configers import config_D

import tensorflow as tf
import numpy as np

class discriminator(object):
    def __init__(self, config, scope_name="discriminator", training=True):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.keep_prob = config.keep_prob
            self.batch_size = config.batch_size

            max_len = config.max_len
            self.input_data = tf.placeholder(tf.int32, [None, max_len])
            self.target = tf.placeholder(tf.int64, [None])
            self.mask_x = tf.placeholder(tf.float32, [max_len, None])

            class_number = config.class_number
            hidden_dim = config.hidden_dim
            vocab_size = config.vocab_size
            embd_dim = config.embd_dim
            hidden_layer_num = config.hidden_layer_num

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                hidden_dim,
                forget_bias=.0,
                state_is_tuple=True
            )
            if self.keep_prob<1.:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell,
                    output_keep_prob=self.keep_prob
                )
            cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell]*hidden_layer_num,
            state_is_tuple=True
            )

            self.initial_state = cells.zero_state(self.batch_size, dtype=tf.float32)

            with tf.device("/cpu:0"), tf.name_scope("embdedding_layer"):
                embedding = tf.get_variable(
                    "embedding",
                    [vocab_size, embd_dim],
                    dtype=tf.float32
                    )
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)

            if self.keep_prob<1.:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            out_put = []
            state = self.initial_state
            with tf.variable_scope("LSTM"):
                for time_step in xrange(max_len):
                    if time_step>0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cells(inputs[:, time_step, :], state)
                    out_put.append(cell_output)
            out_put = out_put * self.mask_x[:, :, None]

            with tf.name_scope("mean_pooling"):
                out_put = tf.reduce_sum(out_put, 0)/(tf.reduce_sum(self.mask_x, 0)[:, None])

            with tf.name_scope("softmax_output"):
                softmax_w = tf.get_variable(
                    "W",
                    [hidden_dim, class_number],
                    dtype=tf.float32
                    )
                softmax_b = tf.get_variable(
                    "b",
                    [class_number],
                    dtype=tf.float32
                )
                self.logits = tf.matmul(out_put, softmax_w) + softmax_b

            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    self.logits+1e-10,
                    self.target
                )
                self.cost = tf.reduce_mean(self.loss)

            with tf.name_scope("accuracy"):
                self.prediction = tf.argmax(self.logits, 1)
                corrects = tf.equal(self.prediction, self.target)
                self.correct_num = tf.reduce_sum(tf.cast(corrects, tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32), name="acc")

            tvars = tf.trainable_variables()
            self.lr = config.lr
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, tvars),
                config.max_grad_norm
            )
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

if __name__ == "__main__":
    D_configure = config_D()
    D = discriminator(D_configure)
