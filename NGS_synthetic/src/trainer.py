#!/usr/bin/env/python
'''
Usage:
    trainer.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       config file path
    --config JSON            config string
    --out PATH               log file path
    --train PATH             training file path(s)
    --valid PATH             validation file path(s)
    --stats PATH             feature statistics file
    --ckpt CKPT              load from checkpoint
    --device DEV             device to run on [default: -1]
    --save_model
'''

from __future__ import print_function
from docopt import docopt
import tensorflow as tf
import numpy as np
import utils as u
import load_data as ld
import time
from dual_biGRU import DualBiGRU



def make_training_ops(basic_loss, hypers):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weights_norm = tf.reduce_mean(tf.stack([tf.nn.l2_loss(w) for w in tf.trainable_variables()]))
    loss = basic_loss + hypers['weight_decay'] * weights_norm
    lr = hypers['learning_rate']

    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads = []
    for grad, var in grads_and_vars:
        if grad is not None:
            clipped_grads.append((tf.clip_by_norm(grad, hypers['clamp_gradient_norm']), var))
        else:
            clipped_grads.append((grad, var))   
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)
    return tf.group(train_op, *update_ops)

def prepare_model(hypers):
    model = DualBiGRU(hypers)
    model.seal()
    model.set_train_op(make_training_ops(model.metrics['loss'], hypers))
    return model

def training_loop(sess, model, data, hypers, is_training, verbose=True):
    loss = accuracy = n_instances = data_ptr = 0
    start_time = time.time()
    
    if not is_training: # feed all data in validation set
        batch_size = len(data) + 1
    else:
        batch_size = hypers['batch_size']

    batch = ld.get_batch(data, data_ptr, data_ptr + batch_size)
    while batch is not None:
        this_batch_size = len(batch['x'])
        n_instances += this_batch_size
        feed_dict = {
            model.inputs['seq']: batch['x'],
            model.inputs['global_features']: np.stack(
                [batch[f] for f in hypers['global_feature_list']],
                axis=1),
            model.metrics['target']: batch['label'],
        }
        if is_training:
            feed_dict[model.inputs['is_training']] = True
            feed_dict[model.inputs['dropout_kp']] = hypers['dropout_kp']
            fetch_list = [model.metrics, model.train_op]
        else:
            feed_dict[model.inputs['is_training']] = False
            feed_dict[model.inputs['dropout_kp']] = 1.0
            fetch_list = [model.metrics, model.outputs]

        result = sess.run(fetch_list, feed_dict)
        loss += result[0]['loss'] * this_batch_size
        accuracy += result[0]['accuracy']

        data_ptr += batch_size
        batch = ld.get_batch(data, data_ptr, data_ptr + batch_size)

    #instance_per_sec = n_instances / (time.time() - start_time)
    accuracy = hypers['stats']['std']['K(obs)'] * np.sqrt(accuracy / n_instances)
    loss = loss / n_instances
    elapsed_time = time.time() - start_time
    if verbose:
       print("loss: %s\t| accuracy: %s\t| instances: %s\t| time: %s" % (loss, accuracy, n_instances, elapsed_time))
    return loss, accuracy, result

def run_epoch(args, model, data, hypers, sess, epoch_id):
    log_entry = {}
    np.random.shuffle(data['train'])

    print_this_epoch = epoch_id % 100 == 0
    show_training = True
    if args['--train'] is not None:
        printing = show_training and print_this_epoch
        if printing:
            print('epoch', epoch_id, 'train ', end='')
        train_loss, train_acc, _ = training_loop(
            sess, model, data['train'], hypers, True, verbose=printing)
        _, _, train_result = training_loop(
            sess, model, data['train'][:hypers['batch_size']], hypers, False, verbose=False)
    else:
        train_loss = train_acc = -1

    if args['--valid'] is not None:
        if print_this_epoch:
            print('epoch', epoch_id, 'valid ', end = '')
        val_loss, val_acc, result = training_loop(
            sess, model, data['valid'], hypers, False, verbose=print_this_epoch)
    else:
        val_loss = val_acc = -1
        result = [[],[]]

    if args['--out'] is not None:
        log_entry['epoch'] = epoch_id
        log_entry['train_loss'] = train_loss;  log_entry['train_acc'] = train_acc
        log_entry['train_result'] = train_result
        log_entry['val_loss'] = val_loss;      log_entry['val_acc'] = val_acc; 
        log_entry['metrics'] = result[0]
        log_entry['outputs'] = result[1]
    return log_entry

def main():
    args = docopt(__doc__)
    hypers = ld.make_hypers(args)
    u.set_random_seed(hypers['seed'])
    
    data = ld.get_all_data(args, hypers)
    hypers['stats'] = ld.load_stats_file(args['--stats'])

    device_id = int(args.get('--device'))
    device_string = u.get_device_string(device_id)
    with tf.device(device_string):
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            model = prepare_model(hypers)
        sess = u.get_tf_session(device_id >= 0)
        checkpoint = args.get('--ckpt')
        saver = u.manage_model_restoration(checkpoint, sess)

        log_to_save = []
        for epoch in range(hypers['n_epochs']):
            log_entry = run_epoch(args, model, data, hypers, sess, epoch)
            log_to_save.append(log_entry)

    u.dump_log(log_to_save, hypers, args.get('--out'))

    if args['--save_model']:
        model_save_path = saver.save(sess, '%s.ckpt' % args['--out'])
        print('model saved to %s' % model_save_path)

    print('done')

if __name__ == "__main__":
    main()