#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:19:22 2018

@author: ldong
"""
#%%
import numpy as np, pandas as pd, glob
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from collections import OrderedDict as od

path = '/workspace/ldong/jigsaw/data/'
files_oof = glob.glob(path+'best/'+'*oof*')
files_oof.sort()

sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission.iloc[:,1::] = 0.0
list_classes = list(sample_submission)[1::]

TRAIN_DATA_FILE=path+'train.csv'
train = pd.read_csv(TRAIN_DATA_FILE)
targets = train.iloc[:,2::]          
    
import cPickle as pk
#with open(path+'best/'+'combined_feats.pkl', 'r') as f:
#    subs_oof, subs = pk.load(f)
with open(path+'best/'+'combined_feats_mc1.pkl', 'r') as f:
    subs_oof, subs = pk.load(f)

import time   
from multiprocessing import Pool 
base_oofs = reduce(lambda x, y: x[list_classes]+y[list_classes], subs_oof.values())/len(subs_oof)
base_score = roc_auc_score(targets, base_oofs)
print 'base score: ', base_score

def mc_opt(buf):
    np.random.seed(buf[0])
    lclass = buf[1]
    n_mc = 500
    
    best_score = 0.0
    best_coeff = np.zeros(len(subs_oof))
    for i_mc in xrange(n_mc):
        rand = np.random.rand(len(subs_oof))
        randn = rand/np.sum(rand) 
    
        oofs = reduce(lambda x, y: x+y, [wi*oofi[lclass] for wi, oofi in zip(randn, subs_oof.values())])
        tmp_score = roc_auc_score(targets[lclass], oofs)
        if tmp_score > best_score:
            best_score = tmp_score
            best_coeff = randn
#            print lclass, ' best score: ', best_score, ' best weights: ', best_coeff
        else:
            continue
    return [best_score, best_coeff]

score_all = 0.
for lclass in list_classes:
    start = time.time()
    ncore = 10
    p = Pool(ncore)
    np.random.seed(233) 
    list_pass = [[i, lclass] for i in np.random.randint(0,10000,ncore)]
    recv = p.map(mc_opt, list_pass)
    best_idx = np.argmax([i[0] for i in recv ])
    best_score = recv[best_idx][0]
    best_coeff = recv[best_idx][1]
    score_all += best_score
    #best_coeff = [0.02106035, 0.0891556  ,0.0108855  ,0.07604912 ,0.13253396 ,0.01180962,\
    # 0.02415087 ,0.15855132 ,0.00935288 ,0.10132532 ,0.07087768 ,0.05935783,\
    # 0.04423592 ,0.11148115 ,0.00594114 ,0.07323173 ]
    #best_oof = reduce ( lambda x,y:x+y, [wi*oofi for wi, oofi in zip(best_coeff, subs_oof.values())] )
    #best_score = roc_auc_score(targets, best_oof) #0.9916293923221611  
    #best_score_classes = od( [('toxic', roc_auc_score(targets.toxic, best_oof.toxic) ),
    #                      ('severe_toxic', roc_auc_score(targets.severe_toxic, best_oof.severe_toxic) ),
    #                      ('obscene', roc_auc_score(targets.obscene, best_oof.obscene) ),
    #                      ('threat', roc_auc_score(targets.threat, best_oof.threat) ),
    #                      ('insult', roc_auc_score(targets.insult, best_oof.insult) ),
    #                      ('identity_hate', roc_auc_score(targets.identity_hate, best_oof.identity_hate) ) ])
    print lclass, ' done, best score: ', best_score
    print 'best weights: ', best_coeff
    sample_submission[lclass] = reduce(lambda x,y: x+y, [wi*subi[lclass] for wi,subi in zip(best_coeff, subs.values())]).values

print 'elapsed time: ', time.time()-start
print 'all class best score: ', score_all/6.

#denom = np.sum([1.0 + 1.5 + 2.0 + 1.75 + 1.25 + 1.1 + 1.0 + 1.1 + 1.1 + 0.8 + 1.5 + 2.0 + 0.8])
#sample_submission[list_classes] = (1.0*l1mx_gru + # GRU 0.9824
#                                   1.0*l1mx_textcnn + # textcnnFastText 0.9834
#                                   1.0*l1mx_mdnn + # 9855
#                                   1.0*l1mx_dpcnn +
#                                   1.0*l1mx_grucnn +
#                                   l1mx_capsule + 
#                                   l1mx_gruclean +
#                                   subs[get_sub_idx('l1m45_mdnnlstmFastText')][list_classes] +
#                                   subs[get_sub_idx('l1m42_lstmFastText')][list_classes] +
#                                   subs[get_sub_idx('l1m44_lstmFastTextcnn')][list_classes] +
#                                   subs[get_sub_idx('l1m43_bilstmNumberbatchen')][list_classes] +
#                                   (subs[get_sub_idx('l1m32_ridge')][list_classes] + subs[get_sub_idx('l1m5_fm')][list_classes] )/2.0 +
#                                   subs[get_sub_idx('l1m4_logit')][list_classes] + 
#                                   subs[get_sub_idx('l1m6_lgb')][list_classes] )/14.0 

## replicate blend-it-all kernel
#lstmnbsmlr = (1.0*subs[get_sub_idx('l1mx_lr9792')][list_classes] + \
#              2.0*(subs[get_sub_idx('l1m7_gruNumberbatchen')][list_classes] + subs[get_sub_idx('l1m1_gruFastText')][list_classes])/2.0 )/3.0
#simpleblend = (subs[get_sub_idx('l1mx_nbsvm9772')][list_classes] + \
#               subs[get_sub_idx('l1mx_nnFastText9736')][list_classes] + \
#               subs[get_sub_idx('l1mx_avenger9825')][list_classes] + \
#               subs[get_sub_idx('l1m27_textcnnNumberbatch')][list_classes] + \
#               (subs[get_sub_idx('l1m2_gruGlove')][list_classes] + subs[get_sub_idx('l1m8_gruNumberbatch')][list_classes])/2.0 )/5.0
#hightblend = (2.0*subs[get_sub_idx('l1m20_dpcnnNumberbatchen')][list_classes] + \
#              1.0*subs[get_sub_idx('l1mx_nnFastText9736')][list_classes] + \
#              2.0*subs[get_sub_idx('l1m3_textcnnFastTextK')][list_classes] + \
#              1.0*( subs[get_sub_idx('l1m25_textcnnGlove')][list_classes] + subs[get_sub_idx('l1m26_textcnnNumberbatchen')][list_classes] )/2.0 + \
#              2.0*subs[get_sub_idx('l1mx_nbsvm9772')][list_classes] + \
#              1.0*subs[get_sub_idx('l1m4_logit')][list_classes] + \
#              2.0*simpleblend[list_classes] + \
#              1.0*lstmnbsmlr[list_classes] + \
#              1.0*subs[get_sub_idx('l1mx_avenger9825')][list_classes] )/13.0
#minlstmnbsvm = (subs[get_sub_idx('l1mx_nbsvm9772')][list_classes] + \
#               (subs[get_sub_idx('l1m15_dpcnnGlove')][list_classes] + subs[get_sub_idx('l1m19_dpcnnFastText')][list_classes])/2.0 )/2.0
#lgbgrulrlstmnbsvm =0.15*subs[get_sub_idx('l1m6_lgb')][list_classes] + \
#                   0.4*subs[get_sub_idx('l1m17_mdnnNumberbatchen')][list_classes]  + \
#                   0.15*subs[get_sub_idx('l1mx_lr9802')][list_classes] + \
#                   0.3*minlstmnbsvm[list_classes]
#onemoreblend = ( 3*subs[get_sub_idx('l1m35_gruFastTextclean')][list_classes] +\
#                 2*subs[get_sub_idx('l1m37_gruNumberbatchenclean')][list_classes] +\
#                 2*subs[get_sub_idx('l1mx_avenger9825')][list_classes] +\
#                 2*subs[get_sub_idx('l1m36_gruGloveclean')][list_classes] +\
#                 4*hightblend[list_classes] )/13.0
#blendblends = 0.5*lgbgrulrlstmnbsvm[list_classes] + 0.5*onemoreblend[list_classes]
#blendall = ( 2.0*(subs[get_sub_idx('l1m28_bigruFastText')][list_classes] + subs[get_sub_idx('l1m29_bigruGlove')][list_classes] + subs[get_sub_idx('l1m30_bigruNumberbatchen')][list_classes] )/3.0 + 
#             2.0*subs[get_sub_idx('l1m16_mdnnFastText')][list_classes] +\
#             4.0*subs[get_sub_idx('l1m9_gruFastTextcnnK')][list_classes] +\
#             1.0*subs[get_sub_idx('l1mx_avenger9825')][list_classes] +\
#             2.0*blendblends[list_classes] +\
#             4.0*hightblend[list_classes] +\
#             2.0*subs[get_sub_idx('l1m5_fm')][list_classes] +\
#             2.0*subs[get_sub_idx('l1m6_lgb')][list_classes] +\
#             1.0*subs[get_sub_idx('l1mx_tidy9788')][list_classes] +\
#             4.0*( subs[get_sub_idx('l1m22_gruGlovecnn')][list_classes] + subs[get_sub_idx('l1m23_gruNumberbatchencnn')][list_classes] )/2.0 +\
#             5.0*subs[get_sub_idx('l1mx_lgbblend9858')][list_classes] )/29.0
#sample_submission[list_classes] = blendall[list_classes]

#lstm with bn


sample_submission.to_csv('/h1/ldong/stacking_simple_mc_combined_feats.csv', index=False)
