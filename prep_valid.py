#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:07:35 2018

@author: ldong
"""

import numpy as np
import pandas as pd
import cPickle as pk
np.random.seed(32)

stratified = False

def get_val(X, y, kfold):
    
    y_single = np.zeros(y.shape[0])
    for i in xrange(y.shape[0]):
        if y[i][0]==1: y_single[i] += 1
        if y[i][1]==1: y_single[i] += 10
        if y[i][2]==1: y_single[i] += 100
        if y[i][3]==1: y_single[i] += 1000
        if y[i][4]==1: y_single[i] += 10000
        if y[i][5]==1: y_single[i] += 100000
        
    from sklearn.model_selection import StratifiedKFold, KFold
    if stratified:
        skf = StratifiedKFold(n_splits=kfold)
    else:
        skf = KFold(n_splits=kfold, shuffle=True)
    val_flag = []
    for _, ival in skf.split(X, y_single):
        tmp_flag = np.zeros(y.shape[0], dtype=bool)
        tmp_flag[ival] = True
        val_flag.append(tmp_flag)
    
    return val_flag

kfold = 10

path = '/workspace/ldong/jigsaw/data/'
TRAIN_DATA_FILE=path+'train.csv'
train = pd.read_csv(TRAIN_DATA_FILE)#.iloc[0:10000]
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
val_flag = get_val(np.zeros(y.shape[0]), y, kfold=kfold)

if stratified:
    with open('data/val_flag_'+str(kfold)+'fold.pkl','w') as f:
        pk.dump(val_flag, f, protocol=pk.HIGHEST_PROTOCOL)
else:
    with open('data/val_flag_'+str(kfold)+'fold_shuffle.pkl','w') as f:
        pk.dump(val_flag, f, protocol=pk.HIGHEST_PROTOCOL)