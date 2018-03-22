#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:19:22 2018

@author: ldong
"""

import numpy as np, pandas as pd, glob
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import xgboost as xgb
import gc
import regex as re

path = '/workspace/ldong/jigsaw/data/'

sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission.iloc[:,1::] = 0.0
list_classes = list(sample_submission)[1::]

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
targets = train[list_classes]

list_sentences_train = train["comment_text"].fillna("_na_").values.tolist()
list_sentences_test = test["comment_text"].fillna("_na_").values.tolist()

def asterix_freq(x):
    return x.count('!')/float(len(x))

def uppercase_freq(x):
    return len(re.findall('[A-Z]',x))/float(len(x))

up_freq_trn = [uppercase_freq(s) for s in list_sentences_train]
up_freq_tst = [uppercase_freq(s) for s in list_sentences_test]
exclaim_freq_trn = [asterix_freq(s) for s in list_sentences_train]
exclaim_freq_tst = [asterix_freq(s) for s in list_sentences_test]

files_oof = glob.glob(path+'best/'+'*oof*')
files_oof.sort()
#files_oof.pop(0) # remove nbsvm

aucs = []
class_oof = pd.DataFrame()   
subs_oof = []
for f in files_oof:
    col = f.split('/')[-1].split('.')[0].split('_')[0] + '_' + f.split('/')[-1].split('.')[0].split('_')[1]
    subs_oof.append(pd.read_csv(f))
    score = roc_auc_score(train.iloc[:,2::],  subs_oof[-1][list_classes])
    print col, ' ', score
    aucs.append(score)
    
    class_oof[col+'_'+list_classes[0]] = subs_oof[-1][list_classes[0]]
    class_oof[col+'_'+list_classes[1]] = subs_oof[-1][list_classes[1]]
    class_oof[col+'_'+list_classes[2]] = subs_oof[-1][list_classes[2]]
    class_oof[col+'_'+list_classes[3]] = subs_oof[-1][list_classes[3]]
    class_oof[col+'_'+list_classes[4]] = subs_oof[-1][list_classes[4]]
    class_oof[col+'_'+list_classes[5]] = subs_oof[-1][list_classes[5]]
    
del subs_oof
gc.collect()
aucs = np.array(aucs)
    
files_sub = glob.glob(path+'best/'+'*submission_10FoldAvg.csv')
files_sub.sort()
#files_sub.pop(0) # remove nbsvm

class_ = pd.DataFrame()  
subs = []
for f in files_sub:
    col = f.split('/')[-1].split('.')[0].split('_')[0] + '_' + f.split('/')[-1].split('.')[0].split('_')[1]
    subs.append(pd.read_csv(f))
    class_[col+'_'+list_classes[0]] = subs[-1][list_classes[0]]
    class_[col+'_'+list_classes[1]] = subs[-1][list_classes[1]]
    class_[col+'_'+list_classes[2]] = subs[-1][list_classes[2]]
    class_[col+'_'+list_classes[3]] = subs[-1][list_classes[3]]
    class_[col+'_'+list_classes[4]] = subs[-1][list_classes[4]]
    class_[col+'_'+list_classes[5]] = subs[-1][list_classes[5]]
del subs
gc.collect()

import cPickle as pk
with open(path+'val_flag_10fold.pkl','r') as f:
    val_flag = pk.load(f)
    
    
#%%
def get_sub_idx(sub):
    list_subs = list(class_oof)
    idx = [i for i in xrange(len(list_subs)) if sub in list_subs[i]]
    return int(idx[0])/6
def get_coeff(sub):
    denom = np.sum([aucs[get_sub_idx(isub)] for isub in sub ])
    coeff = [aucs[get_sub_idx(isub)]/denom for isub in sub]
    return coeff
def avg_feats(feats, sub, name):
    w = get_coeff(sub)
    for ifeats in feats:
        for iclass in list_classes:
            ifeats[name+iclass] = reduce(lambda x, y: x+y, [w[i]*ifeats[sub[i]+'_'+iclass] for i in range(len(w))])
        for isub in sub:
            ifeats.drop([c for c in ifeats.columns if isub in c], axis=1, inplace=True)
#    return feats
#def clean_feats(feats):
#    w = get_coeff(['l1m1_gruFastText','l1m2_gruGlove','l1m7_gruNumberbatchen','l1m8_gruNumberbatch'])
#    feats['l1mx_gru_toxic'] = w[0]*feats.l1m1_gruFastText_toxic + \
#                                   w[1]*feats.l1m2_gruGlove_toxic + \
#                                   w[2]*feats.l1m7_gruNumberbatchen_toxic + \
#                                   w[3]*feats.l1m8_gruNumberbatch_toxic
#                                  
#    feats['l1mx_gru_severe_toxic'] = (feats.l1m1_gruFastText_severe_toxic + 
#                                   feats.l1m2_gruGlove_severe_toxic +
#                                   feats.l1m7_gruNumberbatchen_severe_toxic +
#                                   feats.l1m8_gruNumberbatch_severe_toxic)/4.0
#    feats['l1mx_gru_obscene'] = (feats.l1m1_gruFastText_obscene + 
#                                   feats.l1m2_gruGlove_obscene +
#                                   feats.l1m7_gruNumberbatchen_obscene +
#                                   feats.l1m8_gruNumberbatch_obscene)/4.0
#    feats['l1mx_gru_threat'] = (feats.l1m1_gruFastText_threat + 
#                                   feats.l1m2_gruGlove_threat +
#                                   feats.l1m7_gruNumberbatchen_threat +
#                                   feats.l1m8_gruNumberbatch_threat)/4.0
#    feats['l1mx_gru_insult'] = (feats.l1m1_gruFastText_insult + 
#                                   feats.l1m2_gruGlove_insult +
#                                   feats.l1m7_gruNumberbatchen_insult +
#                                   feats.l1m8_gruNumberbatch_insult)/4.0
#    feats['l1mx_gru_identity_hate'] = (feats.l1m1_gruFastText_identity_hate + 
#                                   feats.l1m2_gruGlove_identity_hate +
#                                   feats.l1m7_gruNumberbatchen_identity_hate +
#                                   feats.l1m8_gruNumberbatch_identity_hate)/4.0
#    feats.drop([c for c in feats.columns if 'l1m1_' in c], axis=1, inplace=True)
#    feats.drop([c for c in feats.columns if 'l1m2_' in c], axis=1, inplace=True)
#    feats.drop([c for c in feats.columns if 'l1m7_' in c], axis=1, inplace=True)
#    feats.drop([c for c in feats.columns if 'l1m8_' in c], axis=1, inplace=True)
#    return feats
#avg_feats([class_oof, class_], ['l1m1_gruFastText','l1m2_gruGlove','l1m7_gruNumberbatchen','l1m8_gruNumberbatch'], 'l1mx_gru_')
#avg_feats([class_oof, class_], ['l1m28_bigruFastText','l1m29_bigruGlove','l1m30_bigruNumberbatchen'], 'l1mx_bigru_')
class_oof['uppercase_freq'] = up_freq_trn
class_oof['exclaimation_freq'] = exclaim_freq_trn
class_['uppercase_freq'] = up_freq_tst
class_['exclaimation_freq'] = exclaim_freq_tst
neg_inv_pos = [float(np.sum(targets[iclass]==0))/np.sum(targets[iclass]==1) for iclass in list_classes]
#%%
list_classes = list(sample_submission)[1::]
np.random.seed(233)
ncore = 48
param1 = {}
param1['booster'] = 'gbtree'
param1['objective'] = 'binary:logistic'
param1['eval_metric'] = 'auc'
param1['eta'] = 0.05
param1['gamma'] = 0
param1['max_depth'] = 4
param1['min_child_weight'] = 1.0
param1['max_delta_step'] = 0.0
param1['silent'] = 1
param1['nthread'] = ncore 
param1['scale_pos_weight'] = neg_inv_pos[0]
param1['subsample'] = 0.9 #0.9
param1['colsample_bylevel'] = 0.95
param1['colsample_bytree'] = 0.8
param1['alpha'] = 0
param1['lambda'] = 3000

param2 = {}
param2['booster'] = 'gbtree'
param2['objective'] = 'binary:logistic'
param2['eval_metric'] = 'auc'
param2['eta'] = 0.05
param2['gamma'] = 0
param2['max_depth'] = 4
param2['min_child_weight'] = 1.0
param2['max_delta_step'] = 0.0
param2['silent'] = 1
param2['nthread'] = ncore 
param2['scale_pos_weight'] = neg_inv_pos[1]
param2['subsample'] = 0.85
param2['colsample_bylevel'] = 0.95
param2['colsample_bytree'] = 0.9
param2['alpha'] = 0
param2['lambda'] = 3000

param3 = {}
param3['booster'] = 'gbtree'
param3['objective'] = 'binary:logistic'
param3['eval_metric'] = 'auc'
param3['eta'] = 0.05
param3['gamma'] = 0
param3['max_depth'] = 4
param3['min_child_weight'] = 1.0
param3['max_delta_step'] = 0.0
param3['silent'] = 1
param3['nthread'] = ncore 
param3['scale_pos_weight'] = neg_inv_pos[2]
param3['subsample'] = 0.85
param3['colsample_bylevel'] = 0.95
param3['colsample_bytree'] = 0.9
param3['alpha'] = 0
param3['lambda'] = 3000

param4 = {}
param4['booster'] = 'gbtree'
param4['objective'] = 'binary:logistic'
param4['eval_metric'] = 'auc'
param4['eta'] = 0.05
param4['gamma'] = 0
param4['max_depth'] = 4
param4['min_child_weight'] = 1.0
param4['max_delta_step'] = 0.0
param4['silent'] = 1 
param4['nthread'] = ncore 
param4['scale_pos_weight'] = neg_inv_pos[3]
param4['subsample'] = 0.65
param4['colsample_bylevel'] = 0.95
param4['colsample_bytree'] = 0.9
param4['alpha'] = 0
param4['lambda'] = 1

param5 = {}
param5['booster'] = 'gbtree'
param5['objective'] = 'binary:logistic'
param5['eval_metric'] = 'auc'
param5['eta'] = 0.05
param5['gamma'] = 1
param5['max_depth'] = 4
param5['min_child_weight'] = 1.0
param5['max_delta_step'] = 0.0
param5['silent'] = 0 
param5['nthread'] = ncore 
param5['scale_pos_weight'] = neg_inv_pos[4]
param5['subsample'] = 0.85
param5['colsample_bylevel'] = 0.95
param5['colsample_bytree'] = 0.9
param5['alpha'] = 0
param5['lambda'] = 3000

param6 = {}
param6['booster'] = 'gbtree'
param6['objective'] = 'binary:logistic'
param6['eval_metric'] = 'auc'
param6['eta'] = 0.05
param6['gamma'] = 0
param6['max_depth'] = 4
param6['min_child_weight'] = 1.0
param6['max_delta_step'] = 0.0
param6['silent'] = 1 
param6['nthread'] = ncore 
param6['scale_pos_weight'] = neg_inv_pos[5]
param6['subsample'] = 0.85
param6['colsample_bylevel'] = 0.95
param6['colsample_bytree'] = 0.9
param6['alpha'] = 0
param6['lambda'] = 3000

param = [param1, param2, param3, param4, param5, param6]
trn_x, trn_y = class_oof, targets
tst_x = class_
history = []
cnt = 0
#list_classes = [list_classes[cnt]]
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True)
for iclass in list_classes:
    sample_submission[iclass] = 0.0
    xg_trn = xgb.DMatrix(trn_x, trn_y[iclass])
    hist = xgb.cv(params=param[cnt], dtrain=xg_trn, num_boost_round=2000, folds=skf, early_stopping_rounds=100, verbose_eval=20, as_pandas=True)
    history.append([hist.shape[0], hist.tail(1).values[0][0]])
    print hist.iloc[-1,:]
    for trn_idx, tst_idx in skf.split(trn_x, trn_y[iclass]):
        xg_trn_tmp = xgb.DMatrix(trn_x.iloc[trn_idx,:], trn_y[iclass][trn_idx])
        xg_tst_tmp = xgb.DMatrix(trn_x.iloc[tst_idx,:], trn_y[iclass][tst_idx])
        bst = xgb.train(params=param[cnt], dtrain=xg_trn_tmp,  evals=[(xg_trn_tmp,'trn'),(xg_tst_tmp,'tst')], num_boost_round=hist.shape[0], verbose_eval=False)
        sample_submission[iclass] += bst.predict(xgb.DMatrix(tst_x), ntree_limit=bst.best_ntree_limit)
    sample_submission[iclass] /= 10
    cnt += 1

print 'mean auc = ', np.mean(np.array(history)[:,1])
sample_submission.to_csv('/h1/ldong/stacking_xgb.csv', index=False)

