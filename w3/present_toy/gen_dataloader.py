import numpy as np
import json
# import simplejson

data_path = "text_data/text.txt"

class Gen_Data_loader(object):
    '''
    Generating batches from the text file. Note that there is no need to
    parse the original JSON. Many thanks to sudongqi:
    https://github.com/sudongqi/TensorFlow_ChatBot

    '''
    def __init__(self, sequence_len=10, batch_size=128):
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, file_path):
        whole = []
        with open(file_path, "r") as f:
            for line in f:
                pair = json.loads(line)
                for QA in pair:
                    for wd_idx in QA:
                        whole.append(wd_idx)
        f.close()
        # print len(whole)
        whole = whole[:200000]

        with open("text_data/partial_real.txt", "w") as f:
            seq_list = []
            for idx, wd in enumerate(whole):
                seq_list.append(wd)
                if (idx+1) % self.sequence_len == 0:
                    # print seq_list
                    connected = " ".join([str(x) for x in seq_list]) + "\n"
                    f.write(connected)
                    seq_list = []
        f.close()
        for idx, word in enumerate(whole):
            if idx % self.sequence_len == 0:
                self.token_stream.append(whole[idx:idx+self.sequence_len])

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        print "There are %d of batches created ..." % self.num_batch
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        # print "ret.shape:", np.transpose(ret).shape
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

class Likelihood_Data_loader(Gen_Data_loader):
    #@overrides
    def __init__(self, batch_size=128, sequence_len=10):
        Gen_Data_loader.__init__(self, batch_size, sequence_len)
        self.likeli_flag = 1

if __name__ == "__main__":
    obj = Gen_Data_loader()
    obj.create_batches(data_path)
    obj.next_batch()
