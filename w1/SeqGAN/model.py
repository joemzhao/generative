import tensorflow.python.ops.tensor_array_ops
import tensorflow.python.ops.control_flow_ops
import tensorflow as tf

class LSTM(object):
    def __init__(self, num_emb, batch_size, emb_dim,
                hidden_dim, sequence_len, start_token,
                learning_rate=0.01, rwd_gamma=0.95):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.rwd_gamma = rwd_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.
        self.grad_clip = 5.
        self.exp_rwd = tf.Variable(tf.zeros([self.sequence_len]))

        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            # recurrent unit, from h_t_minus_1 to h_t
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
            # output unit, from h_t to o_t
            self.g_output_unit = self.create_output_unit(self.g_params)

        # input placeholder, sequences are index of true data, not include start token
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])

        # placeholder for rwd from D or rollout policy
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_len])


    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_recurrent_unit(self, params):
        # creat recurrent units, three gates i, f, o, c_
        # input, forget, output then update memory cell
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))

        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_t_m_1):
            prev_hidden_state, c_prev = tf.unpack(hidden_memory_t_m_1)

            # Input gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(prev_hidden_state, self.Ui) + self.bi
            )

            # Forget gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(prev_hidden_state, self.Uf) + self.bf
            )

            # Output gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(prev_hidden_state, self.Uog) + self.bog
            )

            # New memo cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(prev_hidden_state, self.Uc) + self.bc
            )

            # refresh memo cell
            c = f * c_prev + i * c_

            # current hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.pack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memo_tuple):
            hidden_state, c_prev = tf.unpack(hidden_memo_tuple)
            # hidden_state : batch_size x hidden_dim
            logists = tf.matmul(hidden_state, self.Wo) + self.bo
            # output tf.nn.softmax(logists)
            return logists

        return unit





















#
