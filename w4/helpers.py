import numpy as np
import os
import json

def create_buckets(list_bucket_size):
    '''
    return tuples containing the size of buckets. Note the +1 from the GO signal
    for decoding.
    '''
    buckets = []
    for i in xrange(len(list_bucket_size)):
        for j in xrange(len(list_bucket_size)):
            buckets.append((list_bucket_size[i], list_bucket_size[j]+1))
    return buckets

def data_processing(data, size, batch_size):
    enc_len = size[0]
    dec_len = size[1]

    enc_inp = np.zeros((enc_len, batch_size))
    dec_inp = np.zeros((dec_len, batch_size))
    dec_tar = np.zeros((dec_len, batch_size))

    for i in xrange(len(data)):
        pair = data[i]
        enc_inp[enc_len-len(pair[0]):enc_len, i] = pair[0][::-1]
        dec_inp[1:len(pair[1])+1, i] = pair[1]
        dec_tar[0:len(pair[1]), i] = pair[1]
        # start and end token
        dec_inp[0, i] = 2
        dec_tar[len(pair[1]), i] = 3

    return enc_inp, dec_inp, dec_tar
