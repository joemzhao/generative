import numpy as np
import tensorflow as tf
import helpers
import sys

class naive_predictors(object):
    '''
    A naive predictor containing two predicting methods:
        weighted pick
        argmax

    These are used to generate candidates A' together with the
    beam search predictor
    '''
    def __init__(self, probs, enc_inp, dec_states, encoder_decoder,
                                     max_seq_len, sess, signal=None):
        self.probs = probs
        self.enc_inp = enc_inp
        self.dec_states = dec_states
        self.encoder_decoder = encoder_decoder
        self.max_seq_len = max_seq_len
        self.sess = sess
        self.signal = signal

    def arg_max(self):
        '''
        naive argmax to find the token with highest probability after softmax
        when pick one, set corresponding probaility to -1
        '''
        sequence = [2] # starting token
        if self.signal != None:
            sequence.append(signal)
        dec_inp = helpers.build_input(sequence)
        candidates = []

        feed_dict = {
        self.encoder_decoder.enc_inputs: self.enc_inp,
        self.encoder_decoder.dec_inputs: dec_inp
        }
        prob_list = self.sess.run(self.probs, feed_dict)

        while len(candidates)<self.max_seq_len:
            candidates.append(np.argmax(prob_list))
            prob_list[0][np.argmax(prob_list)] = -1

        return candidates

    def weighted_pick(self):
        '''
        add randomness to the pick-up procedure. softer than argmax
        '''
        sequence = [2] # starting token
        if self.signal != None:
            sequence.append(signal)
        dec_inp = helpers.build_input(sequence)
        candidates = []

        feed_dict = {
        self.encoder_decoder.enc_inputs: self.enc_inp,
        self.encoder_decoder.dec_inputs: dec_inp
        }
        prob_list = self.sess.run(self.probs, feed_dict)

        cdf = np.cumsum(prob_list)
        while len(candidates)<self.max_seq_len:
            idx = int(np.searchsorted(cdf, np.random.rand(1) * 1.))
            while idx in candidates:
                idx = int(np.searchsorted(cdf, np.random.rand(1) * 1.))
            candidates.append(idx)

        return candidates
