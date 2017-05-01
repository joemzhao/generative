import numpy as np
import tensorflow as tf
import fuse_utils as fu

class Fuser(object):
    '''
    An class providing fusing operation and other interfaces with LSTM in seqgan
    '''
    def __init__(self, force=True, batch_size=1, num_batchs=1, candidate_size=21,
                vocab_size=20525, emb_dim=256, hidden_dim=64):
        self.batch_size = batch_size
        self.num_batchs = num_batchs
        self.candidate_size = candidate_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.init_embedding, _, self.cand_beam, _ = fu.load_data(force)
        candidates, self.candidate_max_length = fu.build_up_candidates(self.cand_beam)
        print "Finish loading data for initializing."
        self.Q_batches, self.A_batches, self.GA_batches = fu.build_batches(candidates, self.batch_size, self.num_batchs)
        print "Finish spliting batches."

        self.input_ph = tf.placeholder(tf.int32,
                             [self.batch_size, self.candidate_size, self.candidate_max_length],
                             name="input_candidate")

        self.embedding_ph = tf.placeholder(
                              tf.float32,
                              self.init_embedding.shape,
                              name="embedding_matrix")


    def fuse(self):
        vocab_size, embedding_size = self.embedding_ph.get_shape().as_list()
        batch_size, candidate_size, candidate_length = self.input_ph.get_shape().as_list()
        embedded_input = tf.nn.embedding_lookup(self.embedding_ph, self.input_ph)

        # print('vocab size:{},embedding size:{}'.format(vocab_size, embedding_size))
        # print('batch size:{},candidate size:{}'.format(batch_size, candidate_size))
        # print('size of embedded_input: {}'.format(embedded_input.get_shape()))

        # convolution
        conv_spatials = [2, 2]
        conv_depths = [32, 32]
        archits = zip(conv_spatials, conv_depths)

        pooled_outputs = []
        for i, (conv_spatial, conv_depth) in enumerate(archits):
            with tf.variable_scope("conv-lrelu-pooling-%s"%i):
                W = self.weight_variable(shape=[1, conv_spatial, embedding_size, conv_depth])
                # print "This is parameter W:"
                # print W
                b = self.bias_variable(shape=[conv_depth])
                conv = self.conv2d(
                            embedded_input,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID")
                h = self.leakyrelu(conv, b, alpha=1.0/5.5)
                pooled = self.max_pool(
                            h,
                            ksize=[1, 1, candidate_length-conv_spatial+1, 1],
                            strides=[1, 1, 1, 1],
                            padding="VALID")
                pooled_outputs.append(pooled)
        num_filters_total = sum([x[1] for x in archits])
        features = tf.reshape(
                             tf.concat(pooled_outputs, 3),
                             [batch_size, candidate_size, num_filters_total])
        # print('size of fused feature: {}'.format(features.get_shape()))

        features = tf.transpose(features, [0, 2, 1])
        # print('size of feature for step 1: {}'.format(features.get_shape()))
        features = tf.reshape(features, shape=[batch_size * num_filters_total, -1])
        # print('size of feature for step 2: {}'.format(features.get_shape()))

        W = tf.get_variable(
            "W",
            shape=[candidate_size, 1],
            initializer=tf.contrib.layers.xavier_initializer())

        weighted_features = tf.matmul(features, W)
        final_features = tf.reshape(weighted_features, [batch_size, num_filters_total])

        print('size of the final feature: {}'.format(final_features.get_shape()))
        return final_features # batch_size x 64



    def get_candidates_tofeed(self):
        '''
        Currently for testing the gradient backpropagation.
        '''
        return np.squeeze(np.array(self.GA_batches), axis=(0,))

    def fuse_step(self):
        return self.fuse()
        # sess.run([fused], feed_dict={self.input_ph: input_candidate, self.embedding_ph:emb})


    def weight_variable(self, shape, initmethod=tf.truncated_normal, name="W", trainable=True):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name="b"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

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
        '''
        use relu as activation
        '''
        temp = tf.nn.bias_add(conv, b)
        return tf.maximum(temp * alpha, temp)

    def test_feed(self, sess):
        candidates_tofeed = self.get_candidates_tofeed()
        sess.run([self.input_ph], feed_dict=
                                    {self.input_ph: candidates_tofeed,
                                    self.embedding_ph: self.init_embedding
                                    }
                                )
        print "session run finished!"

if __name__ == "__main__":
    f = Fuser()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    f.test_feed()
