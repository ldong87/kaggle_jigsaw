#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:59:33 2018

@author: ldong
"""

import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import cPickle as pk


path = '/workspace/ldong/jigsaw/data/'
output_prefix = path+'l1m12_nbsvm'#path+sys.argv[0].split('.')[0]
ifold = 0#int(sys.argv[1])
kfold = 10#int(sys.argv[2])

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
subm = pd.read_csv(path+'sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import cPickle as pk
with open(path+'val_flag_'+str(kfold)+'fold.pkl','r') as f:
    val_flag = pk.load(f)
train_xy, valid_xy = train.iloc[~val_flag[ifold],:], train.iloc[val_flag[ifold],:]

import re, string
re_tok = re.compile('([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

#vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#               smooth_idf=1, sublinear_tf=1 )
#trn_term_doc = vec.fit_transform(train_xy[COMMENT])
#val_term_doc = vec.transform(valid_xy[COMMENT])
#test_term_doc = vec.transform(test[COMMENT])
#
#with open(path+'train_test_logit.pkl', 'w') as f:
#    pk.dump([trn_term_doc, val_term_doc, test_term_doc], f, protocol=pk.HIGHEST_PROTOCOL)

with open(path+'train_test_logit.pkl', 'r') as f:
    trn_term_doc, val_term_doc, test_term_doc = pk.load(f)
    
x = trn_term_doc
val_x = val_term_doc
test_x = test_term_doc

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y,i):
#    all_parameters = {
#                  'C'             : [1.048113, 0.1930, 0.596362, 0.25595, 0.449843, 0.25595],
#                  'tol'           : [0.1, 0.1, 0.046416, 0.0215443, 0.1, 0.01],
#                  'solver'        : ['lbfgs', 'newton-cg', 'lbfgs', 'newton-cg', 'newton-cg', 'lbfgs'],
#                  'fit_intercept' : [True, True, True, True, True, True],
#                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
#                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
#                 }
#    
#    m = LogisticRegression(
#        C=all_parameters['C'][i],
#        max_iter=500,
#        tol=all_parameters['tol'][i],
#        solver=all_parameters['solver'][i],
#        fit_intercept=all_parameters['fit_intercept'][i],
#        penalty=all_parameters['penalty'][i],
#        dual=False,
#        class_weight=all_parameters['class_weight'][i],
#        verbose=0,
#        n_jobs=-1)
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3, dual=True, max_iter=500, n_jobs=-1)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(test), len(label_cols)))
preds_val = np.zeros((len(valid_xy), len(label_cols)))
history = {'val_auc':0}
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train_xy[j],i)
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
    preds_val[:,i] = m.predict_proba(val_x.multiply(r))[:,1]
    
from sklearn.metrics import roc_auc_score
history['val_auc'] = roc_auc_score(valid_xy[label_cols], preds_val)
print 'val_auc = ', history['val_auc']
with open(output_prefix+'_y_val_'+str(ifold)+'fold.pkl','w') as f:
    pk.dump(preds_val, f, protocol=pk.HIGHEST_PROTOCOL)
            
submid = pd.DataFrame({'id': subm["id"]})
sample_submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
sample_submission.to_csv(output_prefix+'_submission_fold'+str(ifold)+'.csv', index=False)
import cPickle as pk
with open(output_prefix+'_history'+str(ifold)+'.pkl', 'w') as f:
    pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)