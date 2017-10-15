import numpy as np
import json
import simplejson
import cPickle

class trimed_g_loader(object):
    def __init__(self, batch_size, largest_len, candidates_array):
        self.batch_size = batch_size
        self.largest_len = largest_len
        self.num_batch = 21
        self.candidates = np.array(candidates_array)
        self.pointer = 0

    def this_batch(self):
        return self.candidates[self.pointer-1]

    def next_batch(self):
        ret = self.candidates[self.pointer]
        self.pointer = (self.pointer + 1) % 21
        return ret

    def reset_pointer(self):
        self.pointer = 0

class g_data_loader(object):
    '''
    A dataloader for generator. The inputs are sentences and will be parsed to
    the same length (largest sequence length).
    '''
    def __init__(self, batch_size, largest_len, data_path):
        self.batch_size = batch_size
        self.largest_len = largest_len
        self.data_path = data_path

    def create_batches(self):
        self.sentences_stream = []
        f = open(self.data_path, "r")
        whole = simplejson.load(f)
        temp = []
        for item in whole:
            if len(temp) == self.largest_len:
                self.sentences_stream.append(temp)
                temp = []
            else:
                temp.append(item)
        f.close()
        self.num_batch = int(len(self.sentences_stream) / self.batch_size)
        self.sentences_stream = self.sentences_stream[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(np.array(self.sentences_stream), self.num_batch, 0)
        self.pointer = 0
        print "%d batch created for pretraining!" % self.num_batch

    def this_batch(self):
        return self.sentences_batches[self.pointer-1]

    def next_batch(self):
        ret = self.sentences_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

class full_loader(object):
    def __init__(self):
        self.idx = 33
        out_name = "./seperator/seps/pair_"+str(self.idx)+".p"
        self.Q, self.A, self.candidates = cPickle.load(open(out_name, "r"))

    def pad_candidates(self):
        # max_len = max([len(can) for can in self.candidates])
        max_len = len(self.A)
        temp = []
        for can in self.candidates:
            can = can + [20520] * (max_len-len(can))
            temp.append(can)
        self.candidates = temp
        return self.Q, self.A, self.candidates, max_len, self.idx

if __name__ == "__main__":
    floader = full_loader()
    q, a, candidates, maxlen = floader.pad_candidates()
    print q
    print a
    print np.array(candidates)
    print np.array(candidates).shape
