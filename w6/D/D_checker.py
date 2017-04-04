import tensorflow as tf
import numpy as np
import os
import shutil

from runable_checker_loader import dataloader
from D import LSTM_

BATCH_SIZE = 128
LR = 0.01
EBD_DIM = 128
HID_DIM = 128
NUM_LAY = 1
KEEP_PROB = 0.8
VOCAB_SIZE = 24992
SEQ_LEN = 20

true = '../datasets/checking_D/true_D.txt'
fake = '../datasets/checking_D/fake_D.txt'

reader = dataloader(true, fake, BATCH_SIZE, VOCAB_SIZE, SEQ_LEN)
model_name = "_h"+str(HID_DIM)+"_bs"+str(BATCH_SIZE)+"_x"+"NUM_LAY"
save_path = "save/"+model_name

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

nn = LSTM_(EBD_DIM, HID_DIM, VOCAB_SIZE, BATCH_SIZE, LR, NUM_LAY, KEEP_PROB, SEQ_LEN)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print "NN model build."

count = 0
epoch_loss = 0.
epoch_count = 0
EPOCH = 2
losses = []
saver = tf.train.Saver()

for it in xrange(EPOCH*reader.num_batches):
    sequence_, idx_ = reader.next_batch()
    _mask = sequence_.copy()
    _mask[_mask>0] = 1

    feed_dic = {
        nn.input_data: sequence_,
        nn.target: idx_,
        nn.mask_x: _mask.T
    }

    del _mask

    _, avg_loss, acu = sess.run([nn.train_op, nn.cost, nn.accuracy], feed_dic)
    epoch_loss += avg_loss

    count += 1
    epoch_count += 1

    if count % reader.num_batches == 0:
        print "loss: "+str(avg_loss)+" accu "+str(acu)+" @ iter_idx: "+str(it)+" count: "+ str(epoch_count * BATCH_SIZE)

print "===== finished training the model ===== "
print "Start testing..."
reader.create_test_batch()

count = 0
epoch_count = 0
for t_batch in xrange(reader.num_test_batch):
    sequence_, idx_ = reader.next_test_batch()
    _mask = sequence_.copy()
    _mask[_mask>0] = 1
    feed_dic = {
        nn.input_data: sequence_,
        nn.target: idx_,
        nn.mask_x: _mask.T
    }

    del _mask

    avg_loss, acu = sess.run([nn.cost, nn.accuracy], feed_dic)
    epoch_loss += avg_loss

    count += 1
    epoch_count += 1

    if count % 10 == 0:
        print "loss: "+str(avg_loss)+" accu "+str(acu)+" @ test_batch: "+str(t_batch)+" count: "+ str(epoch_count * BATCH_SIZE)


sess.close()
