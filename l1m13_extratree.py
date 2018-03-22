#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:54:53 2018

@author: ldong
"""

import numpy as np
import pandas as pd
from sklearn import *
from textblob import TextBlob
import sys
import cPickle as pk

path = '/workspace/ldong/jigsaw/data/'
output_prefix = path+'l1m13_extratree'#path+sys.argv[0].split('.')[0]
ifold = 0#int(sys.argv[1])
kfold = 10#int(sys.argv[2])

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

coly = [c for c in train.columns if c not in ['id','comment_text']]
y = train[coly]
tid = test['id'].values

train['polarity'] = train['comment_text'].map(lambda x: int(TextBlob(x.decode('utf-8')).sentiment.polarity * 10))
test['polarity'] = test['comment_text'].map(lambda x: int(TextBlob(x.decode('utf-8')).sentiment.polarity * 10))

train['comment_text'] = train.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
test['comment_text'] = test.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow = train.shape[0]

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=800000)
data = tfidf.fit_transform(df)

with open(path+'val_flag_'+str(kfold)+'fold_shuffle.pkl','r') as f:
    val_flag = pk.load(f)
ind_val = np.where(val_flag[ifold])[0].tolist()
ind_trn = np.where(~val_flag[ifold])[0].tolist()
np.random.shuffle(ind_trn)

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
#model.fit(data[:nrow], y)
model.fit(data[ind_trn], y.iloc[ind_trn])
#print(1- model.score(data[:nrow], y))
preds = model.predict_proba(data[nrow:])
preds_val = model.predict_proba(data[ind_val])
y_val = pd.DataFrame([[c[1] for c in preds_val[row]] for row in range(len(preds_val))]).T
sub2 = pd.DataFrame([[c[1] for c in preds[row]] for row in range(len(preds))]).T
sub2.columns = coly
sub2['id'] = tid
for c in coly:
    sub2[c] = sub2[c].clip(0+1e12, 1-1e12)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
subm = pd.read_csv(path+'sample_submission.csv')
subm[label_cols] = sub2[label_cols]
    
history = {'val_auc':0}
from sklearn.metrics import roc_auc_score
history['val_auc'] = roc_auc_score(y.iloc[ind_val], y_val)
print 'val_auc = ', history['val_auc']
with open(output_prefix+'_y_val_'+str(ifold)+'fold.pkl','w') as f:
    pk.dump(preds_val, f, protocol=pk.HIGHEST_PROTOCOL)
            
subm.to_csv(output_prefix+'_submission_fold'+str(ifold)+'.csv', index=False)
import cPickle as pk
with open(output_prefix+'_history'+str(ifold)+'.pkl', 'w') as f:
    pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)