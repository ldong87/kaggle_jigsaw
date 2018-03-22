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

#aucs = {}
#for f in files_oof:
#    oof_i = pd.read_csv(f)
#    score = roc_auc_score(targets, oof_i)
#    print f.split('/')[-1].split('.')[0], ' ', score
#    aucs.append(score)
#aucs = np.array(aucs)

files_sub = glob.glob(path+'best/'+'*submission_10FoldAvg.csv')
files_sub.sort()
#files_l1mx = glob.glob(path+'best/'+'l1mx_*.csv')
#files_sub.extend(files_l1mx)

class1 = pd.DataFrame()
class2 = pd.DataFrame()
class3 = pd.DataFrame()
class4 = pd.DataFrame()
class5 = pd.DataFrame()
class6 = pd.DataFrame()    
subs = od()
for f in files_sub:
    col = f.split('/')[-1].split('.')[0].split('_')[0] + '_' + f.split('/')[-1].split('.')[0].split('_')[1]
    drop_list = ['l1m12_nbsvm']#, 'l1m32_ridge']
    if col in drop_list:
        print 'Drop ', col
        continue
    subs[col] = pd.read_csv(f)
    subs[col].set_index('id',inplace=True)
    class1[col] = subs[col].toxic
    class2[col] = subs[col].severe_toxic
    class3[col] = subs[col].obscene
    class4[col] = subs[col].threat
    class5[col] = subs[col].insult
    class6[col] = subs[col].identity_hate
    subs[col][list_classes] = minmax_scale(subs[col][list_classes])
#    subs[-1][list_classes] = (subs[-1][list_classes] - subs[-1][list_classes].mean())/(subs[-1][list_classes].max() - subs[-1][list_classes].min())
    
class1_oof = pd.DataFrame()
class2_oof = pd.DataFrame()
class3_oof = pd.DataFrame()
class4_oof = pd.DataFrame()
class5_oof = pd.DataFrame()
class6_oof = pd.DataFrame()    
subs_oof = od()
aucs = od()
for f in files_oof:
    col = f.split('/')[-1].split('.')[0].split('_')[0] + '_' + f.split('/')[-1].split('.')[0].split('_')[1]
    drop_list = ['l1m12_nbsvm']#, 'l1m32_ridge']
    if col in drop_list:
        print 'oof Drop ', col
        continue
    subs_oof[col] = pd.read_csv(f)
    score = roc_auc_score(targets, subs_oof[col])
    print col, ' ', score
    aucs[col] = score
    class1_oof[col] = subs_oof[col].toxic
    class2_oof[col] = subs_oof[col].severe_toxic
    class3_oof[col] = subs_oof[col].obscene
    class4_oof[col] = subs_oof[col].threat
    class5_oof[col] = subs_oof[col].insult
    class6_oof[col] = subs_oof[col].identity_hate
    subs_oof[col][list_classes] = minmax_scale(subs_oof[col][list_classes])
#    subs[-1][list_classes] = (subs[-1][list_classes] - subs[-1][list_classes].mean())/(subs[-1][list_classes].max() - subs[-1][list_classes].min())





#%%
def corr_plot(data, method, mask):
    list_classes = list(data)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    data_corr = data.corr( method=method.split('_')[1])
    cax = ax.matshow(np.ma.array(data_corr, mask=mask))
    fig.colorbar(cax)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(list_classes)))
    ax.set_xticklabels(list_classes, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(list_classes)))
    ax.set_yticklabels(list_classes)
    ax.set_title(method)
#    plt.show()
    fig.savefig(path+method+'.png')
    return data_corr

#mask = 1-np.tri(len(files_sub), k=0)
#mask=np.zeros([len(subs), len(subs)]) # unmask
#class1_corr = corr_plot( class1, 'class1_pearson', mask)
#class2_corr = corr_plot( class2, 'class2_pearson', mask)
#class3_corr = corr_plot( class3, 'class3_pearson', mask)
#class4_corr = corr_plot( class4, 'class4_pearson', mask)
#class5_corr = corr_plot( class5, 'class5_pearson', mask)
#class6_corr = corr_plot( class6, 'class6_pearson', mask)

def get_coeff(sub):
    denom = np.sum([aucs[isub] for isub in sub ])
    coeff = [aucs[isub]/denom for isub in sub]
    return coeff

def mc_opt_general(randseed, list_feats, base_coeff, l1mx_tmp, lclass):
    np.random.seed(randseed)
    n_mc = num_mc
    subs_oof_tmp = od()
    for k, v in subs_oof.iteritems():
        if k in list_feats:
            subs_oof_tmp[k] = v
            
    best_score = roc_auc_score(targets[lclass], l1mx_tmp[lclass])
    best_score_all = roc_auc_score(targets[list_classes], l1mx_tmp[list_classes])
    best_coeff = np.array(base_coeff)
    print lclass, ' Before: best score: ', best_score, ' all best score', best_score_all, ' best weights: ', best_coeff
    for i_mc in xrange(n_mc):
        rand = np.random.rand(len(subs_oof_tmp))
        randn = rand/np.sum(rand) 
    
        best_oofs = reduce(lambda x, y: x+y, [wi*oofi for wi, oofi in zip(randn, subs_oof_tmp.values() )])
        tmp_score = roc_auc_score(targets[lclass], best_oofs[lclass])
        if tmp_score > best_score:
            best_score = tmp_score
            best_coeff = randn
            l1mx_tmp[lclass] = best_oofs[lclass]
#            print 'best score: ', best_score, ' best weights: ', best_coeff
        else:
            continue
    return [best_score, best_coeff, l1mx_tmp]

def combine_feats(subs, isoof, dict_coeffs, lclass):
    def combine_helper(list_feats, subs, feat_name, best_coeff):
        if isoof: 
            coeff = get_coeff(list_feats)
            l1mx = reduce(lambda x,y: x+y, [coeff[i]*subs[list_feats[i]] for i in xrange(len(list_feats))])
            print feat_name                  
            [best_score, best_coeff, l1mx] = mc_opt_general(7, list_feats, coeff, l1mx, lclass)
            best_score_all = roc_auc_score(targets[list_classes], l1mx[list_classes])
            print lclass, ' After: best score: ', best_score, ' all best score', best_score_all, ' best weights: ', best_coeff
            return l1mx, best_coeff
        else:
            coeff = best_coeff['l1mx_'+feat_name]
            l1mx = reduce(lambda x,y: x+y, [coeff[i]*subs[list_feats[i]] for i in xrange(len(list_feats))])
            return l1mx, coeff
        
    # capsule
    if isoof:
        l1mx_capsule, coeff_capsule  = combine_helper( ['l1m34_capsuleFastText', 'l1m39_capsuleGlove', 'l1m40_capsuleNumberbatchen'], subs, 'capsule', [])
        dict_coeffs['l1mx_capsule'] = coeff_capsule
    else:
        l1mx_capsule, coeff_capsule  = combine_helper( ['l1m34_capsuleFastText', 'l1m39_capsuleGlove', 'l1m40_capsuleNumberbatchen'], subs, 'capsule', dict_coeffs)
    # gruclean
    if isoof:
        l1mx_gruclean, coeff_gruclean = combine_helper( ['l1m35_gruFastTextclean', 'l1m36_gruGloveclean', 'l1m37_gruNumberbatchenclean'], subs, 'gruclean', [])
        dict_coeffs['l1mx_gruclean'] = coeff_gruclean
    else:
        l1mx_gruclean, coeff_gruclean = combine_helper( ['l1m35_gruFastTextclean', 'l1m36_gruGloveclean', 'l1m37_gruNumberbatchenclean'], subs, 'gruclean', dict_coeffs)
                                     
    # bigru
    if isoof:
        l1mx_bigru, coeff_bigru = combine_helper( ['l1m28_bigruFastText', 'l1m29_bigruGlove', 'l1m30_bigruNumberbatchen'], subs, 'bigru', [])
        dict_coeffs['l1mx_bigru'] = coeff_bigru        
    else:          
        l1mx_bigru, coeff_bigru = combine_helper( ['l1m28_bigruFastText', 'l1m29_bigruGlove', 'l1m30_bigruNumberbatchen'], subs, 'bigru', dict_coeffs)        
                                         
    # mdnn
    if isoof:
        l1mx_mdnn, coeff_mdnn = combine_helper( ['l1m14_mdnnGlove', 'l1m16_mdnnFastText', 'l1m17_mdnnNumberbatchen'], subs, 'mdnn', [])
        dict_coeffs['l1mx_mdnn'] = coeff_mdnn
    else:            
        l1mx_mdnn, coeff_mdnn = combine_helper( ['l1m14_mdnnGlove', 'l1m16_mdnnFastText', 'l1m17_mdnnNumberbatchen'], subs, 'mdnn', dict_coeffs)        
    
    #dpcnn
    if isoof:
        l1mx_dpcnn, coeff_dpcnn = combine_helper( ['l1m15_dpcnnGlove', 'l1m19_dpcnnFastText', 'l1m20_dpcnnNumberbatchen'], subs, 'dpcnn', [])
        dict_coeffs['l1mx_dpcnn'] = coeff_dpcnn
    else:
        l1mx_dpcnn, coeff_dpcnn = combine_helper( ['l1m15_dpcnnGlove', 'l1m19_dpcnnFastText', 'l1m20_dpcnnNumberbatchen'], subs, 'dpcnn', dict_coeffs)                
 
    # grucnn
    if isoof:
        l1mx_grucnn, coeff_grucnn = combine_helper( ['l1m22_gruGlovecnn', 'l1m9_gruFastTextcnnK', 'l1m23_gruNumberbatchencnn'], subs, 'grucnn', [])                    
        dict_coeffs['l1mx_grucnn'] = coeff_grucnn
    else:
        l1mx_grucnn, coeff_grucnn = combine_helper( ['l1m22_gruGlovecnn', 'l1m9_gruFastTextcnnK', 'l1m23_gruNumberbatchencnn'], subs, 'grucnn', dict_coeffs)                    
        
    # textcnn
    if isoof:
        l1mx_textcnn, coeff_textcnn = combine_helper( ['l1m3_textcnnFastTextK', 'l1m25_textcnnGlove', 'l1m26_textcnnNumberbatchen', 'l1m27_textcnnNumberbatch'], subs, 'textcnn', [])
        dict_coeffs['l1mx_textcnn'] = coeff_textcnn
    else:
        l1mx_textcnn, coeff_textcnn = combine_helper( ['l1m3_textcnnFastTextK', 'l1m25_textcnnGlove', 'l1m26_textcnnNumberbatchen', 'l1m27_textcnnNumberbatch'], subs, 'textcnn', dict_coeffs)                        
   
    # gru
    if isoof:
        l1mx_gru, coeff_gru = combine_helper( ['l1m1_gruFastText', 'l1m2_gruGlove', 'l1m7_gruNumberbatchen', 'l1m8_gruNumberbatch'], subs, 'gru', [])
        dict_coeffs['l1mx_gru'] = coeff_gru
    else:
        l1mx_gru, coeff_gru = combine_helper( ['l1m1_gruFastText', 'l1m2_gruGlove', 'l1m7_gruNumberbatchen', 'l1m8_gruNumberbatch'], subs, 'gru', dict_coeffs)                                    
    
    drop_idx = ['l1m34_capsuleFastText', 'l1m39_capsuleGlove', 'l1m40_capsuleNumberbatchen', \
                                         'l1m35_gruFastTextclean', 'l1m36_gruGloveclean', 'l1m37_gruNumberbatchenclean',\
                                         'l1m28_bigruFastText', 'l1m29_bigruGlove', 'l1m30_bigruNumberbatchen',\
                                         'l1m14_mdnnGlove', 'l1m16_mdnnFastText', 'l1m17_mdnnNumberbatchen',\
                                         'l1m15_dpcnnGlove', 'l1m19_dpcnnFastText', 'l1m20_dpcnnNumberbatchen',\
                                         'l1m22_gruGlovecnn', 'l1m9_gruFastTextcnnK', 'l1m23_gruNumberbatchencnn',\
                                         'l1m3_textcnnFastTextK', 'l1m25_textcnnGlove', 'l1m26_textcnnNumberbatchen', 'l1m27_textcnnNumberbatch',\
                                         'l1m1_gruFastText', 'l1m2_gruGlove', 'l1m7_gruNumberbatchen', 'l1m8_gruNumberbatch']  
    if lclass == 'identity_hate' or type(lclass)==list:
        for i in drop_idx:
            subs.pop(i, 0)
    adds = ['l1mx_capsule', 'l1mx_gruclean', 'l1mx_bigru', 'l1mx_mdnn', 'l1mx_dpcnn', 'l1mx_grucnn', 'l1mx_textcnn', 'l1mx_gru']
    for i in adds:
        try:
            subs[i][lclass] = eval(i)[lclass]
        except:
            subs[i] = eval(i) # when there is no lclass key in subs, meaning it's the first, i.e. toxic
    return subs, dict_coeffs

import time, sys
num_mc = int(sys.argv[1])
start = time.time()
#lclass = list_classes
for lclass in list_classes:
    subs_oof, dict_coeffs = combine_feats(subs_oof, 1, od(), lclass)   
    subs, dict_coeffs = combine_feats(subs, 0, dict_coeffs, lclass)
print 'elapsed time: ', time.time() -start

import cPickle as pk
with open(path+'best/'+'combined_feats_mc'+str(num_mc)+'.pkl', 'w') as f:
    pk.dump([subs_oof, subs], f, protocol=pk.HIGHEST_PROTOCOL)