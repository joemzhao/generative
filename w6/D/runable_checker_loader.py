import numpy as np
import os
import json

class dataloader(object):
    def __init__(self, true, fake, batch_size=128, vocab_size=24991, sequence_len=20):
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.true = true
        self.fake = fake
        self.get_data_and_label()

    def get_data_and_label(self):
        '''
        Read all sentences and corresponding labels from BBT and opensubtitle dataset (trimed).
        33431 sentences in total.
        Padding is used if the sequence length is smaller than
        20
        '''
        self.all_sentence = []
        self.all_label = []

        with open(self.true) as t:
            for line in t:
                line = json.loads(line)
                while len(line[0])<20:
                    line[0].append(0)
                self.all_sentence.append(line[0])
                self.all_label.append(line[1])
        t.close()

        with open(self.fake) as f:
            for line in f:
                line = json.loads(line)
                while len(line[0])<20:
                    line[0].append(0)
                self.all_sentence.append(line[0])
                self.all_label.append(line[1])
        f.close()
        self.spliting()

    def spliting(self):
        '''
        Spliting the full data to train and test data sets.
        0.8 * 33431 are training data, 0.2 * 33431 are testing.
        The data is shuffled first.
        '''
        self.all_label = np.array(self.all_label)
        self.all_sentence = np.array(self.all_sentence)

        shuffle_indices = np.random.permutation(np.arange(len(self.all_label)))
        self.all_label = self.all_label[shuffle_indices]
        self.all_sentence = self.all_sentence[shuffle_indices]

        bound = int(0.8 * 33431)
        self.train_sentence = self.all_sentence[:bound]
        self.train_label = self.all_label[:bound]

        self.test_sentence = self.all_sentence[bound:]
        self.test_label = self.all_label[bound:]

        self.create_batches()

    def create_batches(self):
        '''
        Creating batches using training data
        '''
        self.num_batches = int(len(self.train_label)/self.batch_size)
        print "There are %d batches created!" % self.num_batches

        self.train_sentence = self.train_sentence[:self.num_batches*self.batch_size]
        self.train_label = self.train_label[:self.num_batches*self.batch_size]

        self.sequence_batch = np.split(self.train_sentence, self.num_batches, 0)
        self.label_batch = np.split(self.train_label, self.num_batches, 0)
        self.pointer = 0

    def next_batch(self):
        ret_seq = self.sequence_batch[self.pointer]
        ret_lab = self.label_batch[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batches
        return ret_seq, ret_lab

    def reset_pointer(self):
        self.pointer = 0

    def create_test_batch(self):
        self.num_test_batch = int(len(self.test_label)/self.batch_size)
        print "There are %d batches created for test the model!" % self.num_test_batch

        self.test_sentence = self.test_sentence[:self.num_test_batch*self.batch_size]
        self.test_label = self.test_label[:self.num_test_batch*self.batch_size]

        self.sequence_test_batch = np.split(self.test_sentence, self.num_test_batch, 0)
        self.label_test_batch = np.split(self.test_label, self.num_test_batch, 0)
        self.test_pointer = 0

    def next_test_batch(self):
        ret_seq = self.sequence_test_batch[self.test_pointer]
        ret_lab = self.label_test_batch[self.test_pointer]
        self.test_pointer = (self.test_pointer + 1) % self.num_test_batch
        return ret_seq, ret_lab

    def reset_test_pointer(self):
        self.test_pointer = 0



if __name__ == "__main__":
    true = '../datasets/checking_D/true_D.txt'
    fake = '../datasets/checking_D/fake_D.txt'
    obj = dataloader(true=true, fake=fake, batch_size=128)
    obj.next_batch()
