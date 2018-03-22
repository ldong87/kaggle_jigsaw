#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 18:53:16 2018

@author: ldong
"""

import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "48"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, PReLU, add
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
import cPickle as pk

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class GetBest(Callback):
    def __init__(self, trn_data, val_data, val_flag=True,
                 monitor='val_acc', verbose=0, mode='max', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        
        self.x = trn_data[0]
        self.y = trn_data[1]
        self.x_val = val_data[0]
        self.y_val = val_data[1]
        self.aucs = []
        self.val_aucs = []
        
        self.val_flag = val_flag

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
            
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)
        
        if self.val_flag == True:
            y_pred = self.model.predict(self.x)
            roc = roc_auc_score(self.y, y_pred)
            y_pred_val = self.model.predict(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val)
            print('\rroc-auc: %s - roc-auc_val: %s \n' % (str(round(roc,4)),str(round(roc_val,4))))
            self.aucs.append(roc)
            self.val_aucs.append(roc_val)
        else:
            y_pred = self.model.predict(self.x)
            roc = roc_auc_score(self.y, y_pred)
            print('\rroc-auc: %s \n' % (str(round(roc,4))))
            self.aucs.append(roc)

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            
path = '/workspace/ldong/jigsaw/data/'
output_prefix = path+sys.argv[0].split('.')[0]
ifold = int(sys.argv[1])
kfold = int(sys.argv[2])
    
train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")
#embedding_path = path+"crawl-300d-2M.vec"
embedding_path = path+"glove.840B.300d.txt"
embed_size = 300
max_features = 130000
max_len = 200

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
train["comment_text"].fillna("no comment")
test["comment_text"].fillna("no comment")

with open(path+'clean_data.pkl', 'r') as f:
    train, test = pk.load(f)

#X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

with open(path+'val_flag_'+str(kfold)+'fold.pkl','r') as f:
    val_flag = pk.load(f)
ind_trn = np.where(~val_flag[ifold])[0].tolist()
np.random.shuffle(ind_trn)
X_train, X_valid, Y_train, Y_valid = train.iloc[ind_trn,:], train.iloc[val_flag[ifold],:], y[ind_trn], y[val_flag[ifold]]

raw_text_train = X_train["comment_text"].str.lower()
raw_text_valid = X_valid["comment_text"].str.lower()
raw_text_test = test["comment_text"].str.lower()

tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text_train)
X_train["comment_seq"] = tk.texts_to_sequences(raw_text_train)
X_valid["comment_seq"] = tk.texts_to_sequences(raw_text_valid)
test["comment_seq"] = tk.texts_to_sequences(raw_text_test)

X_train = pad_sequences(X_train.comment_seq, maxlen = max_len)
X_valid = pad_sequences(X_valid.comment_seq, maxlen = max_len)
test = pad_sequences(test.comment_seq, maxlen = max_len)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization

callback = GetBest(trn_data=(X_train,Y_train), val_data=(X_valid,Y_valid), monitor='val_acc', verbose=1, mode='max', period=1)
early_stop = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=1)

def build_model():
    #wrote out all the blocks instead of looping for simplicity
    filter_nr = 64
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 256
    spatial_dropout = 0.2
    dense_dropout = 0.1
    train_embed = False
    
    inp = Input(shape=(max_len,))
    emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(inp)
    emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)
    
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_comment)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    
    #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment)
    resize_emb = PReLU()(resize_emb)
        
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)
    
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
        
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)
    
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
        
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)
    
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    
    output = add([block4, block3_output])
    max1d = GlobalMaxPooling1D()(output)
    avg1d = GlobalAveragePooling1D()(output)
    output = concatenate([max1d, avg1d])
#    output = Dense(dense_nr, activation='linear')(output)
#    output = BatchNormalization()(output)
#    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(6, activation='sigmoid')(output)

    model = Model(inputs = inp, outputs = output)
    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ["accuracy"])
#    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    History = model.fit(X_train, Y_train, batch_size = 128, epochs = 24, validation_data = (X_valid, Y_valid), 
                        verbose = 1, callbacks = [callback, early_stop])
    return model, History

model, History = build_model()
pred = model.predict(test, batch_size = 1024, verbose = 1)
y_val  = model.predict(X_valid, batch_size=1024, verbose=1)
with open(output_prefix+'_y_val_'+str(ifold)+'fold.pkl','w') as f:
    pk.dump(y_val, f, protocol=pk.HIGHEST_PROTOCOL)
    
sample_submission = pd.read_csv(path+'sample_submission.csv')#.iloc[0:10000]
sample_submission[list_classes] = pred

history = History.history
history['val_auc'] = callback.val_aucs

sample_submission.to_csv(output_prefix+'_submission_fold'+str(ifold)+'.csv', index=False)
import cPickle as pk
with open(output_prefix+'_history'+str(ifold)+'.pkl', 'w') as f:
    pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)
    