import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.sparse import hstack
import time
import regex as re
import string
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import lightgbm as lgb
from collections import defaultdict
import gc
from sklearn.metrics import roc_auc_score
import cPickle as pk
    

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print name + ' done in ', time.time() - t0 , 's'
 
 
if __name__ == '__main__':

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    path = '/workspace/ldong/jigsaw/data/'

    with timer("Reading input files"):
        train = pd.read_csv(path+'train.csv').fillna(' ')
        test = pd.read_csv(path+'test.csv').fillna(' ')


    with open(path+'fm_train_test_feat.pkl', 'r') as f:
        train_features, test_features = pk.load(f)
        
    import sys
    ifold = int(sys.argv[1])
    kfold = int(sys.argv[2])
    output_prefix = path+sys.argv[0].split('.')[0]
    with open(path+'val_flag_'+str(kfold)+'fold.pkl','r') as f:
        val_flag = pk.load(f)
    
    print("Shapes just to be sure : ", train_features.shape, test_features.shape)
    
    # Set LGBM parameters
    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 48,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1
    }
    
    val_pred = train[class_names].iloc[val_flag[ifold],:]
    with timer("Scoring lgbm"):
        history = {'val_auc':[]}
        submission = pd.DataFrame.from_dict({'id': test['id']})
        trn_lgbset = lgb.Dataset(train_features, free_raw_data=False)
        
        for i_c, class_name in enumerate(class_names):
            class_pred = np.zeros(len(val_pred))
            train_target = train[class_name]
            trn_lgbset.set_label(train_target.values)
            submission[class_name] = 0.0
            lgb_rounds = 500
            watchlist = [trn_lgbset.subset(np.where(~val_flag[ifold])[0].tolist()), 
                         trn_lgbset.subset(np.where( val_flag[ifold])[0].tolist()) ]
            
            model = lgb.train(
                    params=params,
                    train_set=watchlist[0],
                    num_boost_round=lgb_rounds,
                    valid_sets=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=1
                )
            class_pred = model.predict(trn_lgbset.data[np.where( val_flag[ifold])[0].tolist()], 
                                                       num_iteration=model.best_iteration)
            score = roc_auc_score(train_target.values[val_flag[ifold]], class_pred)
            
            print('CV score for class %-15s is full %.6f ' % (class_name, score) )
            val_pred[class_name] = class_pred
            
            with timer("Predicting probabilities for %s" % class_name):
                submission[class_name] = model.predict(test_features, num_iteration=model.best_iteration)                
                
        cv_score = roc_auc_score(train[class_names].iloc[val_flag[ifold],:].values, val_pred)
        history['val_auc'] = cv_score
        print('CV score is full %.6f ' % (cv_score) )
        
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    with open(output_prefix+'_y_val_'+str(ifold)+'fold.pkl','w') as f:
        pk.dump(val_pred[list_classes].values, f, protocol=pk.HIGHEST_PROTOCOL)
    
    with open(output_prefix+'_history'+str(ifold)+'.pkl', 'w') as f:
        pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)        
        
#        train[["id"] + class_names + [f + "_oof" for f in class_names]].to_csv("lvl0_wordbatch_clean_oof.csv",
#                                                                               index=False,
#                                                                               float_format="%.8f")

            
    submission.to_csv(output_prefix+'_submission_fold'+str(ifold)+'.csv', index=False)
    print 'lgbm Done!'
