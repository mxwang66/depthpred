#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json

name = 'Yuan_G5'
n_fold = 20
stop_epoch = 1000
folder_list = ['logs']

print('read predictions...')
for folder in folder_list:
    print(folder)
    all_target = []
    all_pred = []
    for i in range(n_fold):
        with open(folder+'/'+name+'_fold'+str(i)+'.json', 'r') as f:
            epoches = json.load(f)
        with open(folder+'/'+name+'_class'+str(i)+'.stats', 'r') as f:
            stats = json.load(f)
        target = epoches[stop_epoch-1]['metrics']['target']
        target = [t*stats['std_depth']+stats['mean_depth'] for t in target]
        all_target += target
        pred = epoches[stop_epoch-1]['outputs']['pred_mean']
        pred = [p*stats['std_depth']+stats['mean_depth'] for p in pred]
        all_pred += pred
        with open((folder+'/'+name+'_fold'+str(i)+'.csv'), 'w') as f:
            for line in range(len(target)):
                f.write('%f,%f\n' % (target[line], pred[line]))

    with open((folder+'/pred.csv'), 'w') as f:
        for ii in range(len(all_target)):
            f.write('%f,%f\n' % (all_target[ii], all_pred[ii]))
    print('finished')

print('read loss...')
name = 'Yuan_G5_fold'
for folder in folder_list:
    print(folder)
    stop_epoch = []
    all_train_loss = []
    all_test_loss = []
    plt.figure(figsize=(9,6))
    for i in range(n_fold):
        target = []
        pred = []
        train_loss = []
        val_loss = []
        with open(folder+'/'+name+str(i)+'.json', 'r') as f:
            epoches = json.load(f)
        for epoch in range(len(epoches)):
            train_loss.append(epoches[epoch]['train_loss'])
            val_loss.append(epoches[epoch]['val_loss'])
        stop_epoch.append(val_loss.index(min(val_loss)))
        plt.plot(range(len(epoches)), train_loss, color='b')
        plt.plot(range(len(epoches)), val_loss, color='r')
        all_train_loss.append(train_loss)
        all_test_loss.append(val_loss)

    plt.axvline(sum(stop_epoch)/n_fold, color='g')
    plt.xlabel('epoch', fontsize = 24, fontname='Arial')
    plt.ylabel('loss', fontsize = 24, fontname='Arial')
    plt.yticks(size = 16, fontname='Arial')
    plt.xticks(size = 16, fontname='Arial')
    print('stop epoch: %f' % (sum(stop_epoch)/n_fold))
    plt.savefig(folder+'/'+'loss_stop=%.0f.eps' % (sum(stop_epoch)/n_fold), dpi=300)
    plt.show()

    with open(folder+'/'+'train_loss.csv', 'w') as f1, open(folder+'/'+'test_loss.csv', 'w') as f2:
        for n_epoch in range(len(all_train_loss[0])):
            f1.write('%f' % all_train_loss[0][n_epoch])
            f2.write('%f' % all_test_loss[0][n_epoch])
            for train_loss in all_train_loss[1:]:
                f1.write(',%f' % train_loss[n_epoch])
            f1.write('\n')
            for test_loss in all_test_loss[1:]:
                f2.write(',%f' % test_loss[n_epoch])
            f2.write('\n')
    print('finished')