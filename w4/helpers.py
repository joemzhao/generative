import numpy as np
import os

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
