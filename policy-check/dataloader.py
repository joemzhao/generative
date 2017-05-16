import numpy as np
import json
import simplejson

class Gen_Data_loader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file, seq_len):
        f = open(data_file, "r")
        whole = simplejson.load(f)
        temp = []
        for item in whole:
            if len(temp) == seq_len:
                self.token_stream.append(temp)
                temp = []
            else:
                temp.append(item)
        f.close()
        print len(whole)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0
        print "%d batch created for pretraining!" % self.num_batch
        return len(whole)

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file, seq_len):
        # Load data
        positive_examples = []
        negative_examples = []
        f = open(positive_file, "r")
        whole = simplejson.load(f)
        temp = []
        for item in whole:
            if len(temp) == seq_len:
                positive_examples.append(temp)
                temp = []
            else:
                temp.append(item)
        f.close()

        with open(negative_file) as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == seq_len:
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0
        return self.num_batch


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

if __name__ == "__main__":
    real_path = "./datasets/bbt_concate_full.txt"
    nega_path = "./save/generated.txt"
    g = Gen_Data_loader(128)
    g.create_batches(real_path, 20)
