import os
import json
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data_file(file_name, hypers):
    full_path = os.path.join(file_name)
    print("loading data from: ", full_path)
    with open(full_path, 'r') as f:
        data = json.load(f)
    n_nodes = max([len(g['node_features']) for g in data])
    data_instances = to_instances(data, n_nodes, hypers)
    x_dim = len(data_instances[0]['x'][0])
    metrics = {
        'n_nodes': n_nodes,
        'x_dim'  : x_dim,
    }
    print(metrics)
    return data_instances, metrics

def load_stats_file(file_name):
    with open(file_name, 'r') as f:
        stats = json.load(f)
    return stats

def get_all_data(args, hypers):
    data = {'train': [], 'valid': []}
    for section in data.keys():
        file_paths = args['--%s' % section]
        if file_paths is not None:
            print ('%s data:' % section)
            for file_path in file_paths.split(','):
                d, metrics = load_data_file(file_path.strip(), hypers)
                data[section] += d
        print("%i %s instances loaded" % (len(data[section]), section))
    hypers.update(metrics)
    return data

def to_instances(data, n_nodes, hypers):
    instances = []

    conversion_matrix = [
        [1,0],  #   A: is_purine,  !makes_3_H_bonds
        [0,1],  #   C: !is_purine, makes_3_H_bonds
        [0,0],  #   T: !is_purine, !makes_3_H_bonds
        [1,1]   #   G: is_purine,  makes_3_H_bonds
    ]

    for d in data:
        
        seq_mask = np.array([hypers['sequence_mask'], hypers['sequence_mask'], hypers['toehold_bit_mask'], hypers['open_prob_mask'], hypers['protected_prob_mask']]).reshape([1,-1])
        
        f = np.array(d['node_features'])
        #new_f = np.concatenate([np.matmul(np.array(f[:,:4]), conversion_matrix), f[:,4:-1]], axis=1)
        new_f = np.concatenate([np.matmul(np.array(f[:,:4]), conversion_matrix), f[:,4:]], axis=1)
        masked_f = new_f * seq_mask
        instance_dict = {
            'x'          : masked_f,
            'label'      : d['global_features']['K(obs)'],
        }
        instance_dict.update({feat: hypers['global_feature_mask'][feat] * d['global_features'][feat] for feat in hypers['global_feature_list']})
        instances.append(instance_dict)
    return instances


def default_hypers():
    with open('default_hypers.json', 'r') as f:
        hypers = json.load(f)
    return hypers

def make_hypers(args):
    hypers = default_hypers()
    config_file = args.get('--config-file')
    if config_file is not None:
        with open(config_file, 'r') as f:
            hypers.update(json.load(f))
    config = args.get('--config')
    if config is not None:
        hypers.update(json.loads(config))
    return hypers

def get_batch(data, start, stop):
    if start > len(data):
        return None
    if stop > len(data):
        stop = len(data)
    elements = data[start:stop]
    batch = {k:[] for k in data[0].keys()}
    for d in elements:
        for k in batch.keys():
            batch[k].append(d[k])
    return batch

