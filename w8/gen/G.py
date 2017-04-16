from __future__ import absolute_import, division
from six.moves import xrange

import random
import sys
import tensorflow as tf
import numpy as np

import seq2seq as rl_seq2seq
import data_utils

class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, max_grad_norm, batch_size, learning_rate, learning_rate_decay_factor, use_lstm=False, num_samples=512, forward_only=False, scope_name="gen_seq2seq", dtype=tf.float32):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.buckets = buckets
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)
            self.batch_size = batch_size

            self.up_reward = tf.placeholder(tf.bool, name="up_reward")
            self.en_output_proj = tf.placeholder(tf.bool, name="en_output_proj")

            output_projection = None
            softmax_loss_function = None

            def policy_gradient(logit, labels):
                def softmax(x):
                    return tf.exp(x) / tf.reduce_sum(tf.exp(x), reduction_indices=0)
                prob = softmax(logit)
                return tf.reduce_max(prob)

            softmax_loss_function = policy_gradient

            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            cells = single_cell
            if num_layers > 1:
                cells = tf.nn.rnn_cell.MultiRNNCell([single_cell]*num_layers)

            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return rl_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cells,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode,
                    dtype=dtype
                )

            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = [] #the masks
            for i in xrange(buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            for i in xrange(buckets[-1][1]+1): # note the +1 for go
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(dtype, shape=[None], name="decoder{0}".format(i)))
            targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs)-1)] # shift the targets 1

            if forward_only:
                self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    targets,
                    self.target_weights,
                    buckets,
                    lambda x, y: seq2seq_f(x, y, True),
                    softmax_loss_function=softmax_loss_function
                )
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [
                            tf.matmul(output, output_projection[0]) + output_projection[1]
                            for output in self.outputs[b]
                        ]
            else:
                self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    targets,
                    self.target_weights,
                    buckets,
                    lambda x, y: seq2seq_f(x, y, False),
                    softmax_loss_function=softmax_loss_function
                )

            self.tvars = tf.trainable_variables()
            self.gradient_norms = []
            self.updates = []
            self.reward = [tf.placeholder(tf.float32, name="reward_%i" % i) for i in range(len(buckets))]
            opt = tf.train.AdamOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                adjusted_losses = tf.mul(self.losses[b], self.reward[b])
                gradients = tf.gradients(adjusted_losses, self.tvars)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_grad_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, self.tvars), global_step=self.global_step))
            all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
            self.saver = tf.train.Saver(all_variables)
    def step(step, sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True, up_reward=False, reward=None, debug=True):
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {
            self.up_reward: up_reward,
            self.en_output_proj: (forward_only)
        }
        for l in xrange(len(self.buckets)):
            input_feed[self.reward[l].name] = reward if reward else 1
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id], self.gradient_norms[bucket_id], self.losses[bucket_id]]
        else: # RL
            output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id])

        outputs = sess.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None # Gradient norm, loss, not generating
        else:
            return outputs[0], outputs[1], outputs[2:] # encoder_state, loss, generated words

    def get_batch(self, train_data, bucket_id, type=0):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        batch_size = self.batch_size
        #print("Batch_Size: %s" %self.batch_size)
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_source_encoder, batch_source_decoder = [], []
        #print("bucket_id: %s" %bucket_id)
        if type == 1:
            batch_size = 1
        for batch_i in xrange(batch_size):
          if type == 1:
            encoder_input, decoder_input = train_data[bucket_id]
          elif type == 2:
            #print("data[bucket_id]: ", data[bucket_id][0])
            encoder_input_a, decoder_input = train_data[bucket_id][0]
            encoder_input = encoder_input_a[batch_i]
          elif type == 0:
              encoder_input, decoder_input = random.choice(train_data[bucket_id])
              print("train en: %s, de: %s" %(encoder_input, decoder_input))

          batch_source_encoder.append(encoder_input)
          batch_source_decoder.append(decoder_input)
          # Encoder inputs are padded and then reversed.
          encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
          encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

          # Decoder inputs get an extra "GO" symbol, and are padded then.
          decoder_pad_size = decoder_size - len(decoder_input) - 1
          decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
          batch_encoder_inputs.append(
              np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
          batch_decoder_inputs.append(
              np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(batch_size)], dtype=np.int32))

          # Create target_weights to be 0 for targets that are padding.
          batch_weight = np.ones(batch_size, dtype=np.float32)
          for batch_idx in xrange(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
              target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
              batch_weight[batch_idx] = 0.0
          batch_weights.append(batch_weight)

        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder)

if __name__ == "__main__":
    _buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]
    obj = Seq2SeqModel(1000, 1000, _buckets, 128, 1, 5., 128, .1, 0, use_lstm=False, num_samples=512, forward_only=False, scope_name="gen_seq2seq", dtype=tf.float32)
