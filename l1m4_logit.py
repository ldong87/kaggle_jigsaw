import numpy as np
import pandas as pd
import re
import cPickle as pk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

# Data processing was done as in Bojan's fork of the original script:
# https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

path = '/workspace/ldong/jigsaw/data/'

with open(path+'clean_data.pkl', 'r') as f:
    train, test = pk.load(f)

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
traintime = timer(None)
train_time = timer(None)
tr_ids = train[['id']]
target = train[class_names]

train["new_comment_text"] = train['comment_text']
test["new_comment_text"] = test['comment_text']

trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
#del trate, tete
train.drop(["new_comment_text"], axis=1, inplace=True)
test.drop(["new_comment_text"], axis=1, inplace=True)

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
timer(train_time)

#train_time = timer(None)
#print(' Part 1/2 of vectorizing ...')
#word_vectorizer = TfidfVectorizer(
#    sublinear_tf=True,
#    strip_accents='unicode',
#    analyzer='word',
#    token_pattern=r'\w{1,}',
#    stop_words='english',
#    ngram_range=(1, 1),
#    max_features=10000)
#word_vectorizer.fit(all_text)
#train_word_features = word_vectorizer.transform(train_text)
#test_word_features = word_vectorizer.transform(test_text)
#timer(train_time)
#
#train_time = timer(None)
#print(' Part 2/2 of vectorizing ...')
#char_vectorizer = TfidfVectorizer(
#    sublinear_tf=True,
#    strip_accents='unicode',
#    analyzer='char',
#    stop_words='english',
#    ngram_range=(2, 6),
#    max_features=50000)
#char_vectorizer.fit(all_text)
#train_char_features = char_vectorizer.transform(train_text)
#test_char_features = char_vectorizer.transform(test_text)
#timer(train_time)
#
#train_features = hstack([train_char_features, train_word_features]).tocsr()
#test_features = hstack([test_char_features, test_word_features]).tocsr()
#timer(traintime)
#
#with open(path+'logit_train_test_feat.pkl', 'w') as f:
#    pk.dump([train_features,test_features], f, protocol=pk.HIGHEST_PROTOCOL)
    
with open(path+'logit_train_test_feat.pkl', 'r') as f:
    train_features, test_features = pk.load(f)

all_parameters = {
                  'C'             : [1.048113, 0.1930, 0.596362, 0.25595, 0.449843, 0.25595],
                  'tol'           : [0.1, 0.1, 0.046416, 0.0215443, 0.1, 0.01],
                  'solver'        : ['lbfgs', 'newton-cg', 'lbfgs', 'newton-cg', 'newton-cg', 'lbfgs'],
                  'fit_intercept' : [True, True, True, True, True, True],
                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
                 }

submission = pd.DataFrame.from_dict({'id': test['id']})

import sys
ifold = int(sys.argv[1])
kfold = int(sys.argv[2])
with open(path+'val_flag_'+str(kfold)+'fold.pkl','r') as f:
    val_flag = pk.load(f)

idpred = tr_ids

traintime = timer(None)
val_auc = 0.0
val_pred = train.iloc[val_flag[ifold],:]
for j, (class_name) in enumerate(class_names):
#    train_target = train[class_name]

    classifier = LogisticRegression(
        C=all_parameters['C'][j],
        max_iter=200,
        tol=all_parameters['tol'][j],
        solver=all_parameters['solver'][j],
        fit_intercept=all_parameters['fit_intercept'][j],
        penalty=all_parameters['penalty'][j],
        dual=False,
        class_weight=all_parameters['class_weight'][j],
        verbose=0)

    avreal = target[class_name]
    lr_cv_sum = 0
    lr_pred = []
    lr_fpred = []

    train_time = timer(None)
    
    X_train, X_val = train_features[~val_flag[ifold]], train_features[val_flag[ifold]]
    y_train, y_val = target.loc[~val_flag[ifold]], target.loc[val_flag[ifold]]

    classifier.fit(X_train, y_train[class_name])
    y_val_pred = classifier.predict_proba(X_val)[:, 1]
    lr_y_pred = classifier.predict_proba(test_features)[:, 1]
    val_auc += roc_auc_score(y_val[class_name], y_val_pred)
    print('\n Fold %02d class %s val_auc: %.6f' % ((ifold+1), class_name, roc_auc_score(y_val[class_name], y_val_pred)))

    timer(train_time)

    submission[class_name] = lr_y_pred 
    val_pred[class_name] = y_val_pred

output_prefix = path+sys.argv[0].split('.')[0]
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

with open(output_prefix+'_y_val_'+str(ifold)+'fold.pkl','w') as f:
    pk.dump(val_pred[list_classes].values, f, protocol=pk.HIGHEST_PROTOCOL)
            
sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission[list_classes] = submission[list_classes]

history = {'val_auc': val_auc/6.0}

sample_submission.to_csv(output_prefix+'_submission_fold'+str(ifold)+'.csv', index=False)
import cPickle as pk
with open(output_prefix+'_history'+str(ifold)+'.pkl', 'w') as f:
    pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)

    
print 'logit Done!'

timer(traintime)