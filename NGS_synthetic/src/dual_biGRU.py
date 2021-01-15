from model import Model
import tensorflow as tf
import numpy as np
import utils as u

N_OUT = 2

class DualBiGRU(Model):
    def __init__(self, hypers):
        super().__init__(hypers)
        self.make_weights()

    def make_placeholders(self):
        inputs = {
            'seq': tf.placeholder(tf.float32, [None, self.hypers['n_nodes'], self.hypers['x_dim']]),
            'global_features'  : tf.placeholder(tf.float32, [None, len(self.hypers['global_feature_list'])]),
            'is_training': tf.placeholder_with_default(False, None),
            'dropout_kp': tf.placeholder_with_default(1.0, None)
        }
        return inputs

    def make_weights(self):
        output_size_factor = 4 if not self.hypers['symmetric'] else 2
        self.output_MLP = u.MLP(
            output_size_factor*self.hypers['hidden_size'] 
             + len(self.hypers['global_feature_list']),
            N_OUT,
            self.hypers['output_hid_sizes'])
        self.fw_cell = tf.contrib.rnn.GRUCell(self.hypers['hidden_size'], 
            kernel_initializer=tf.glorot_uniform_initializer(seed=self.hypers['seed']))
        if self.hypers["share_fwd_bwd"]:
            self.bw_cell = self.fw_cell
        else:
            self.bw_cell = tf.contrib.rnn.GRUCell(self.hypers['hidden_size'], 
                kernel_initializer=tf.glorot_uniform_initializer(seed=(1+self.hypers['seed'])))
                  

    def __call__(self, inputs, out_fmt='regression'):
        target_seq = inputs['seq'][:,:self.hypers['n_nodes']//2, :]
        probe_seq = inputs['seq'][:,self.hypers['n_nodes']//2:, :]
        with tf.variable_scope("target_GRU"):
            _, rnn_output_states_target = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell, self.bw_cell, target_seq, dtype=tf.float32)
        with tf.variable_scope("probe_GRU"):
            _, rnn_output_states_probe = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell, self.bw_cell, probe_seq, dtype=tf.float32)

        if self.hypers['bidirectional']:
            fw_bw_output = tf.concat(list(rnn_output_states_target) + list(rnn_output_states_probe), 1)
        else:
            print("!WARNING: unidirectional")
            fw_bw_output = tf.concat([rnn_output_states_target[0], 0*rnn_output_states_target[1]] +
                                [0*rnn_output_states_probe[0], rnn_output_states_probe[1]], 1)
        if self.hypers['symmetric']:
            fw_bw_output = tf.concat([rnn_output_states_target[0]+rnn_output_states_probe[0],
                                      rnn_output_states_target[1]+rnn_output_states_probe[1]], 1)
            if not self.hypers['bidirectional']:
                fw_bw_output = tf.concat([rnn_output_states_target[0]+rnn_output_states_probe[0],
                                          0.0*(rnn_output_states_target[1]+rnn_output_states_probe[1])], 1)

        if out_fmt == 'regression':
            return self.regression_output(fw_bw_output, inputs)
        else:
            return fw_bw_output

    def regression_output(self, h, inputs):
        #concat global features
        h = tf.concat([self.hypers['local_feature_weight'] * h, inputs['global_features']], axis=1)
        output = self.output_MLP(h, dropout_keep_prob=inputs['dropout_kp'])
        return {
            'pred_mean': tf.reshape(output[:,0], [-1]),
            'pred_precision': tf.reshape(tf.abs(output[:,1]), [-1])
        }

    def make_metrics(self, pred_mean, pred_precision, hypers):
        target = tf.placeholder(tf.float32, [None])
        diff_sqr = (target - pred_mean)**2
        if hypers['heteroskedastic']:
            loss = tf.reduce_mean(0.5 * diff_sqr * pred_precision - 0.5 * tf.log(pred_precision/(2*np.pi)))
        else:
            loss = tf.reduce_mean(0.5 * diff_sqr)
        accuracy = tf.reduce_sum(diff_sqr)
        return {'loss': loss, 'accuracy': accuracy, 'target': target}

    def seal(self):
        self.inputs = self.make_placeholders()
        self.outputs = self(self.inputs)
        self.metrics = self.make_metrics(
            self.outputs['pred_mean'], self.outputs['pred_precision'], self.hypers)