import numpy as np
import tensorflow as tf
import random
import json
import os, sys

#### file paths
dirs = {
    #'raw' : os.path.join('..', '201711'),
    'raw' : os.path.join('..', '201802'),
    'processed': os.path.join('..', 'processed_201802'),
    'folds': os.path.join('..', 'processed_201802', 'folds'),
    'LOO': os.path.join('..', 'processed_201802', 'LOO'),
    'LOCO': os.path.join('..', 'processed_201802', 'LOCO'),
    'logs': os.path.join('..', 'logs')
}
#     #'raw' : os.path.join('..', '201711'),
#     'raw' : os.path.join('..', '201711'),
#     'processed': os.path.join('..', 'processed'),
#     'folds': os.path.join('..', 'processed', 'folds'),
#     'LOO': os.path.join('..', 'processed', 'LOO'),
#     'LOCO': os.path.join('..', 'processed', 'LOCO'),
#     'logs': os.path.join('..', 'logs')
# }
for d in dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

def get_graphs_filepath(dataset_flag):
    return os.path.join(dirs['processed'], '%s_graphs.json' % dataset_flag)

def get_stats_filepath(dataset_flag):
    return os.path.join(dirs['processed'], '%s_feature_stats.json' % dataset_flag)
    
def get_fold_filepath(dataset_flag, n_folds, fold_id):
    return os.path.join(dirs['folds'], '%s_%i_%i.json' % (dataset_flag, n_folds, fold_id))

def get_LOO_filepath(dataset_flag, id):
    return os.path.join(dirs['LOO'], '%s_%s.json' % (dataset_flag, id))

def get_LOCO_filepath(id):
    return os.path.join(dirs['LOCO'], 'class_%s.json' % (id))
####




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_device_string(device_id):
    return '/cpu:0' if device_id < 0 else '/gpu:%s' % device_id

def get_tf_session(is_gpu):
    config = tf.ConfigProto(allow_soft_placement=True)
    if is_gpu:
        config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess

def manage_model_restoration(checkpoint_file_name, sess):
    saver = tf.train.Saver()
    if checkpoint_file_name is not None:
        print('restoring model from %s' % checkpoint_file_name)
        saver.restore(sess, checkpoint_file_name)
    else:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
    return saver

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)

def dump_log(log_to_save, hypers, file_name):
    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            json.dump(log_to_save, f, indent=4, cls=NumpyEncoder)
        hypers_file = '%s.hypers' % os.path.splitext(file_name)[0]
        with open(hypers_file, 'w') as f:
            json.dump(hypers, f, indent=4)
        print('log dumped to %s' % file_name)


def xavier_init(shape):
    scale = np.sqrt(6.0/(shape[-2] + shape[-1]))
    return scale * (2 * np.random.rand(*shape).astype(np.float32) -1)

def norm_and_drop(acts, is_training=True, dropout_keep_prob=1.0, do_bn=True):
    if do_bn:
        acts = tf.layers.batch_normalization(acts, training=is_training)
    return tf.nn.dropout(acts, dropout_keep_prob)

class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1],dims[1:]))
        bias_sizes = dims[1:]
        weights = [tf.Variable(xavier_init(s)) for s in weight_sizes]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32)) for s in weight_sizes]
        network_params = {"weights" : weights, "biases" : biases}
        return network_params
    
    def __call__(self, inputs, is_training=True, dropout_keep_prob=1.0, do_bn=False):
        acts = inputs
        for W,b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, W) + b
            acts = tf.nn.relu(hid)
            acts = norm_and_drop(acts, is_training=is_training, 
                        dropout_keep_prob=dropout_keep_prob, do_bn=do_bn)
        last_hidden = hid
        return last_hidden

def print_temporary(s):
    print('%s     \r' % s, end='')
    sys.stdout.flush()