import numpy as np
import tensorflow as tf
import trim_fuse_utils as fu

class Fuser(object):
    '''
    An class providing fusing operation and other interfaces with LSTM in seqgan
    '''
    def __init__(self, g_emb, batch_size=1, cand_max_len=20, candidate_size=21, vocab_size=20526, emb_dim=80, hidden_dim=64):
        self.batch_size = batch_size
        self.candidate_size = candidate_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cand_max_len = cand_max_len
        self.g_emb = g_emb

        ''' Placehoders. For fuser there is only one placeholder required. The
            place holders of candidates whose embedding to be fused. The initial
            embedding is actually only used once (for each QA pair).
        '''
        self.input_ph = tf.placeholder(tf.int32, [self.batch_size, self.candidate_size, self.cand_max_len], name="input_candidate")
        self.fuse()

    def fuse(self, reuse=False):
        embedded_input = tf.nn.embedding_lookup(self.g_emb, self.input_ph)
        conv_spatials = [2, 2]
        conv_depths = [32, 32]
        archits = zip(conv_spatials, conv_depths)

        pooled_outputs = []
        for i, (conv_spatial, conv_depth) in enumerate(archits):
            with tf.variable_scope("fuser-conv-lrelu-pooling-%s"%i) as scope:
                if not reuse:
                    _W_ = tf.get_variable(
                        name="_W_",
                        shape=[1, conv_spatial, self.emb_dim, conv_depth],
                        initializer=tf.contrib.layers.xavier_initializer())
                    _b_ = tf.get_variable(
                        name="_b_", shape=[conv_depth],
                        initializer=tf.contrib.layers.xavier_initializer())
                else:
                    scope.reuse_variables()
                    _W_ = tf.get_variable(name="_W_")
                    _b_ = tf.get_variable(name="_b_")

                conv = self.conv2d(
                            embedded_input,
                            _W_,
                            strides=[1, 1, 1, 1],
                            padding="VALID")
                h = self.leakyrelu(conv, _b_, alpha=1.0/5.5)
                pooled = self.max_pool(
                            h,
                            ksize=[1, 1, self.cand_max_len-conv_spatial+1, 1],
                            strides=[1, 1, 1, 1],
                            padding="VALID")
                pooled_outputs.append(pooled)
        num_filters_total = sum([x[1] for x in archits])
        features = tf.reshape(
                             tf.concat(pooled_outputs, 3),
                             [self.batch_size, self.candidate_size, num_filters_total])

        features = tf.transpose(features, [0, 2, 1])
        features = tf.reshape(features, shape=[self.batch_size * num_filters_total, -1])

        with tf.variable_scope("fuser-weight-of-candidates") as scope:
            if not reuse:
                W_ = tf.get_variable(name="W_", shape=[self.candidate_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            else:
                scope.reuse_variables()
                W_ = tf.get_variable("W_")

        weighted_features = tf.matmul(features, W_)
        self.final_features = tf.reshape(weighted_features, [self.batch_size, num_filters_total])

    def weight_variable(self, shape, initmethod=tf.truncated_normal, name="W_CNN", trainable=True):
        return tf.get_variable(shape=shape, name=name)

    def bias_variable(self, shape, name="b"):
        return tf.get_variable(shape=shape, name=name)

    def max_pool(self, x, ksize, strides, padding="SAME", name="pool"):
        '''
        max pooling. x -> [batch, height, width, channels]
        '''
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, name=name)

    def conv2d(self, x, W, strides, padding="SAME", name="conv"):
        '''
        x -> [batch, in_height, in_width, in_channels] treat as image but in_channels=1
        W -> [filter_height, filter_width, in_channels, out_channels]
        '''
        return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)

    def leakyrelu(self, conv, b, alpha=0.01, name="leaky_relu"):
        ''' use relu as activation '''
        temp = tf.nn.bias_add(conv, b)
        return tf.maximum(temp * alpha, temp)

if __name__ == "__main__":
    f = Fuser()
