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

path = '/workspace/ldong/jigsaw/data/'
files_oof = glob.glob(path+'best/'+'*oof*')
files_oof.sort()

sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission.iloc[:,1::] = 0.0
list_classes = list(sample_submission)[1::]

TRAIN_DATA_FILE=path+'train.csv'
train = pd.read_csv(TRAIN_DATA_FILE)

aucs = []
for f in files_oof:
    oof_i = pd.read_csv(f)
    score = roc_auc_score(train.iloc[:,2::], oof_i)
    print f.split('/')[-1].split('.')[0], ' ', score
    aucs.append(score)
aucs = np.array(aucs)

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
subs = []
for f in files_sub:
    col = f.split('/')[-1].split('.')[0].split('_')[0] + '_' + f.split('/')[-1].split('.')[0].split('_')[1]
    drop_list = ['l1m12_nbsvm']#, 'l1m32_ridge']
    if col in drop_list:
        print 'Drop ', col
        continue
    subs.append(pd.read_csv(f))
    class1[col] = subs[-1].toxic
    class2[col] = subs[-1].severe_toxic
    class3[col] = subs[-1].obscene
    class4[col] = subs[-1].threat
    class5[col] = subs[-1].insult
    class6[col] = subs[-1].identity_hate
    subs[-1][list_classes] = minmax_scale(subs[-1][list_classes])
#    subs[-1][list_classes] = (subs[-1][list_classes] - subs[-1][list_classes].mean())/(subs[-1][list_classes].max() - subs[-1][list_classes].min())
    
class1_oof = pd.DataFrame()
class2_oof = pd.DataFrame()
class3_oof = pd.DataFrame()
class4_oof = pd.DataFrame()
class5_oof = pd.DataFrame()
class6_oof = pd.DataFrame()    
subs_oof = []
for f in files_oof:
    col = f.split('/')[-1].split('.')[0].split('_')[0] + '_' + f.split('/')[-1].split('.')[0].split('_')[1]
    drop_list = ['l1m12_nbsvm']#, 'l1m32_ridge']
    if col in drop_list:
        print 'Drop ', col
        continue
    subs_oof.append(pd.read_csv(f))
    class1_oof[col] = subs_oof[-1].toxic
    class2_oof[col] = subs_oof[-1].severe_toxic
    class3_oof[col] = subs_oof[-1].obscene
    class4_oof[col] = subs_oof[-1].threat
    class5_oof[col] = subs_oof[-1].insult
    class6_oof[col] = subs_oof[-1].identity_hate
    subs_oof[-1][list_classes] = minmax_scale(subs_oof[-1][list_classes])
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

def get_sub_idx(sub):
    list_subs = list(class1)
    idx = [i for i in xrange(len(list_subs)) if list_subs[i]==sub]
    return idx[0]
def get_coeff(sub):
    denom = np.sum([aucs[get_sub_idx(isub)] for isub in sub ])
    coeff = [aucs[get_sub_idx(isub)]/denom for isub in sub]
    return coeff
def combine_feats(subs):
    # capsule
    coeff_capsule = get_coeff(['l1m34_capsuleFastText', 'l1m39_capsuleGlove', 'l1m40_capsuleNumberbatchen'])
    l1mx_capsule = coeff_capsule[0]*subs[get_sub_idx('l1m34_capsuleFastText')][list_classes] + \
                                      coeff_capsule[1]*subs[get_sub_idx('l1m39_capsuleGlove')][list_classes] + \
                                      coeff_capsule[2]*subs[get_sub_idx('l1m40_capsuleNumberbatchen')][list_classes]
    # gruclean
    coeff_gruclean = get_coeff(['l1m35_gruFastTextclean', 'l1m36_gruGloveclean', 'l1m37_gruNumberbatchenclean'])
    l1mx_gruclean = coeff_gruclean[0]*subs[get_sub_idx('l1m35_gruFastTextclean')][list_classes] + \
                                      coeff_gruclean[1]*subs[get_sub_idx('l1m36_gruGloveclean')][list_classes] + \
                                      coeff_gruclean[2]*subs[get_sub_idx('l1m37_gruNumberbatchenclean')][list_classes]
    # bigru
    coeff_bigru = get_coeff(['l1m28_bigruFastText', 'l1m29_bigruGlove', 'l1m30_bigruNumberbatchen'])
    l1mx_bigru = coeff_bigru[0]*subs[get_sub_idx('l1m28_bigruFastText')][list_classes] + \
                                      coeff_bigru[1]*subs[get_sub_idx('l1m29_bigruGlove')][list_classes] + \
                                      coeff_bigru[2]*subs[get_sub_idx('l1m30_bigruNumberbatchen')][list_classes]  
    # mdnn
    coeff_mdnn = get_coeff(['l1m14_mdnnGlove', 'l1m16_mdnnFastText', 'l1m17_mdnnNumberbatchen'])
    l1mx_mdnn = coeff_mdnn[0]*subs[get_sub_idx('l1m14_mdnnGlove')][list_classes] + \
                                      coeff_mdnn[1]*subs[get_sub_idx('l1m16_mdnnFastText')][list_classes] + \
                                      coeff_mdnn[2]*subs[get_sub_idx('l1m17_mdnnNumberbatchen')][list_classes]
    #dpcnn
    coeff_dpcnn = get_coeff(['l1m15_dpcnnGlove', 'l1m19_dpcnnFastText', 'l1m20_dpcnnNumberbatchen'])
    l1mx_dpcnn = coeff_dpcnn[0]*subs[get_sub_idx('l1m15_dpcnnGlove')][list_classes] + \
                                      coeff_dpcnn[1]*subs[get_sub_idx('l1m19_dpcnnFastText')][list_classes] +\
                                      coeff_dpcnn[2]*subs[get_sub_idx('l1m20_dpcnnNumberbatchen')][list_classes] 
    # grucnn
    coeff_grucnn = get_coeff(['l1m22_gruGlovecnn', 'l1m9_gruFastTextcnnK', 'l1m23_gruNumberbatchencnn'])
    l1mx_grucnn = coeff_grucnn[0]*subs[get_sub_idx('l1m22_gruGlovecnn')][list_classes] +\
                                      coeff_grucnn[1]*subs[get_sub_idx('l1m9_gruFastTextcnnK')][list_classes] +\
                                      coeff_grucnn[2]*subs[get_sub_idx('l1m23_gruNumberbatchencnn')][list_classes] 
    # textcnn
    coeff_textcnn = get_coeff(['l1m3_textcnnFastTextK', 'l1m25_textcnnGlove', 'l1m26_textcnnNumberbatchen', 'l1m27_textcnnNumberbatch'])
    l1mx_textcnn = coeff_textcnn[0]*subs[get_sub_idx('l1m3_textcnnFastTextK')][list_classes] + \
                                      coeff_textcnn[1]*subs[get_sub_idx('l1m25_textcnnGlove')][list_classes] + \
                                      coeff_textcnn[2]*subs[get_sub_idx('l1m26_textcnnNumberbatchen')][list_classes] + \
                                      coeff_textcnn[3]*subs[get_sub_idx('l1m27_textcnnNumberbatch')][list_classes]     
    # gru
    coeff_gru = get_coeff(['l1m1_gruFastText', 'l1m2_gruGlove', 'l1m7_gruNumberbatchen', 'l1m8_gruNumberbatch'])                  
    l1mx_gru = coeff_gru[0]*subs[get_sub_idx('l1m1_gruFastText')][list_classes] + \
               coeff_gru[1]*subs[get_sub_idx('l1m2_gruGlove')][list_classes] + \
               coeff_gru[2]*subs[get_sub_idx('l1m7_gruNumberbatchen')][list_classes] + \
               coeff_gru[3]*subs[get_sub_idx('l1m8_gruNumberbatch')][list_classes]
    drop_idx = [get_sub_idx(i) for i in ['l1m34_capsuleFastText', 'l1m39_capsuleGlove', 'l1m40_capsuleNumberbatchen', \
                                         'l1m35_gruFastTextclean', 'l1m36_gruGloveclean', 'l1m37_gruNumberbatchenclean',\
                                         'l1m28_bigruFastText', 'l1m29_bigruGlove', 'l1m30_bigruNumberbatchen',\
                                         'l1m14_mdnnGlove', 'l1m16_mdnnFastText', 'l1m17_mdnnNumberbatchen',\
                                         'l1m15_dpcnnGlove', 'l1m19_dpcnnFastText', 'l1m20_dpcnnNumberbatchen',\
                                         'l1m22_gruGlovecnn', 'l1m9_gruFastTextcnnK', 'l1m23_gruNumberbatchencnn',\
                                         'l1m3_textcnnFastTextK', 'l1m25_textcnnGlove', 'l1m26_textcnnNumberbatchen', 'l1m27_textcnnNumberbatch',\
                                         'l1m1_gruFastText', 'l1m2_gruGlove', 'l1m7_gruNumberbatchen', 'l1m8_gruNumberbatch']  ]
    drop_idx.sort(reverse=True)
    print drop_idx
    for i in drop_idx:
        subs.pop(i)
    subs.extend([l1mx_capsule, l1mx_gruclean, l1mx_bigru, l1mx_mdnn, l1mx_dpcnn, l1mx_grucnn, l1mx_textcnn, l1mx_gru])
    return subs

subs = combine_feats(subs)
subs_oof = combine_feats(subs_oof)    
import time   
from multiprocessing import Pool 
targets = train.iloc[:,2::]          
base_oofs = reduce(lambda x, y: x[list_classes]+y[list_classes], subs_oof)/len(subs_oof)
base_score = roc_auc_score(targets, base_oofs)
print 'base score: ', base_score

def mc_opt(randseed):
    np.random.seed(randseed)
    n_mc = 63000
    
    best_score = 0.0
    best_coeff = np.zeros(len(subs_oof))
    for i_mc in xrange(n_mc):
        rand = np.random.rand(len(subs_oof))
        randn = rand/np.sum(rand) #np.array([rand[i,:]/rowsum[i] for i in xrange(rand.shape[0])])
    
        oofs_ = [wi*oofi for wi, oofi in zip(randn, subs_oof)]
        oofs = reduce(lambda x, y: x+y, oofs_)
        tmp_score = roc_auc_score(targets, oofs)
        if tmp_score > best_score:
            best_score = tmp_score
            best_coeff = randn
            print 'best score: ', best_score, ' best weights: ', best_coeff
        else:
            continue
    return [best_score, best_coeff]
#start = time.time()
#p = Pool(30)
#recv = p.map(mc_opt, range(233,263))
#print 'elapsed time: ', time.time()-start
#best_idx = np.argmax([i[0] for i in recv ])
#best_score = recv[best_idx][0]
#best_coeff = recv[best_idx][1]
#print 'best score: ', best_score
#print 'best weights: ', best_coeff
#sample_submission[list_classes] = reduce(lambda x,y: x+y, [wi*subi[list_classes] for wi,subi in zip(best_coeff, subs)])

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


#sample_submission.to_csv('/h1/ldong/stacking_simple_mc_combined_feats.csv', index=False)
