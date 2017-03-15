import tensorflow as tf
import numpy as np

class compressor(object):
    ''' compressing the 4-d tensor along the candidate dimension
        or time dimension.
        NOTE: if pre-padding required, it must be implemented before
        calling method from this compressor.
    '''
    def __init__(self, in_tensor, convolution=True):
        self.in_tensor = in_tensor
        self.B = 1
        self.N = in_tensor.get_shape().as_list()[1]
        self.E = in_tensor.get_shape().as_list()[2]
        self.T = in_tensor.get_shape().as_list()[3]
        self._filter_shape = [self.N, 1, self.T, self.T]

    def _cross_correlate(self, sess):
        _filter = tf.Variable(tf.random_uniform(self._filter_shape,
                                               -1., 1.), name="filters")
        self._output = tf.nn.conv2d(
            self.in_tensor,
            filter=_filter,
            strides=[1, 1, 1, 1],
            padding="VALID"
        )

        sess.run(tf.global_variables_initializer())

    def printer(self, sess, to_print):
        print sess.run(to_print)

if __name__ == "__main__":

    N = 3 # candidates
    E = 5 # embedding dimension
    T = 10 # time dependency dimension

    in_tensor = tf.Variable(tf.random_uniform([1, N, E, T], -1., 1.), name="in")
    ope = compressor(in_tensor)

    sess = tf.InteractiveSession()
    ope._cross_correlate(sess)

    print "This is input tensor: "
    ope.printer(sess, ope.in_tensor)
    print "----------------------------"
    print "This is convolution output:"
    ope.printer(sess, ope._output)
