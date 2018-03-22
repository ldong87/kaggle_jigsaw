#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:15:34 2018

@author: ldong
"""

import sys, numpy as np, pandas as pd
from timeit import default_timer as timer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
num_cores = 32
#from keras import backend 
#backend.set_session(backend.tf.Session(config=backend.tf.ConfigProto(inter_op_parallelism_threads=num_cores,\
#                                                                     intra_op_parallelism_threads=num_cores,\
#                                                                     device_count={'CPU':num_cores})))
import os
os.environ['OMP_NUM_THREADS'] = str(num_cores)

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

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
        
#class roc_callback(Callback):
#    def __init__(self,trn_data,val_data):
#        self.x = trn_data[0]
#        self.y = trn_data[1]
#        self.x_val = val_data[0]
#        self.y_val = val_data[1]
#        self.aucs = []
#        self.val_aucs = []
#
#    def on_train_end(self, logs={}):
#        y_pred = self.model.predict(self.x)
#        roc = roc_auc_score(self.y, y_pred)
#        y_pred_val = self.model.predict(self.x_val)
#        roc_val = roc_auc_score(self.y_val, y_pred_val)
#        print('\rroc-auc: %s - roc-auc_val: %s \n' % (str(round(roc,4)),str(round(roc_val,4))))
#        self.aucs.append(roc)
#        self.val_aucs.append(roc_val)
#        return

if __name__ == "__main__":
    
    output_prefix = 'data/'+sys.argv[0].split('.')[0]

    t0 = timer()
    
    ifold = int(sys.argv[1])
    kfold = int(sys.argv[2])
    if kfold ==0: epoch_median = ifold
    
    path = '/workspace/ldong/jigsaw/data/'
    EMBEDDING_FILE=path+'glove.840B.300d.txt'
    TRAIN_DATA_FILE=path+'train.csv'
    TEST_DATA_FILE=path+'test.csv'
    
    embed_size = 300 # how big is each word vector
    max_features = 30000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 100 # max number of words in a comment to use
    rnn_units = 80
#    dense1 = 80
    dense2 = 6
    batch_size = 32
    
    train = pd.read_csv(TRAIN_DATA_FILE)#.iloc[0:10000]
    test = pd.read_csv(TEST_DATA_FILE)#.iloc[0:10000]
    
    list_sentences_train = train["comment_text"].fillna("_na_").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("_na_").values
    
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    if kfold != 0:
        import cPickle as pk
        with open('data/val_flag_'+str(kfold)+'fold.pkl','r') as f:
            val_flag = pk.load(f)
        trn_x, trn_y = X_t[~val_flag[ifold]], y[~val_flag[ifold]]
        val_x, val_y = X_t[val_flag[ifold]], y[val_flag[ifold]]
    else:
        trn_x, trn_y = X_t, y
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
#    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    embedding_matrix = np.zeros([nb_words, embed_size])
    for word, i in word_index.items():
        if i >= max_features: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(rnn_units, return_sequences=True))(x)
    #                      activation='relu', recurrent_activation='relu',
    #                      kernel_regularizer=regularizers.L1L2(0.1,0.1), recurrent_regularizer=regularizers.L1L2(0.1,0.1),
#                          return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    x = concatenate([avg_pool, max_pool])
#    x = BatchNormalization()(x)
#    x = Dense(dense1, activation="relu")(x)
#    x = Dropout(0.1)(x)
#    x = BatchNormalization()(x)
    x = Dense(dense2, activation="sigmoid")(x)
    
    print 'Timer before Model: ', timer()-t0
        
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=1)
    
    if kfold != 0:
        #    callback_roc = roc_callback(trn_data=(trn_x,trn_y), val_data=(val_x,val_y))
        callback = GetBest(trn_data=(trn_x,trn_y), val_data=(val_x,val_y), 
                           monitor='val_acc', verbose=1, mode='max', period=1)
        History = model.fit(trn_x, trn_y, batch_size=batch_size, epochs=24, 
                            validation_data=(val_x, val_y), callbacks=[callback, early_stop])
        y_val  = model.predict([val_x], batch_size=1024, verbose=1)
        with open(output_prefix+'_y_val_'+str(ifold)+'fold.pkl','w') as f:
            pk.dump(y_val, f, protocol=pk.HIGHEST_PROTOCOL)
    else:
        checkpointer = ModelCheckpoint(filepath='/workspace/ldong/jigsaw/checkpoint/rerun.hdf5', verbose=1)
        callback = GetBest(trn_data=(trn_x,trn_y), val_data=(None, None), val_flag=False,
                           monitor='acc', verbose=1, mode='max', period=1)
        History = model.fit(trn_x, trn_y, batch_size=batch_size, epochs=epoch_median, callbacks=[callback])#[checkpointer, callback])
        
    y_test = model.predict([X_te], batch_size=1024, verbose=1)
    
    sample_submission = pd.read_csv(path+'sample_submission.csv')#.iloc[0:10000]
    sample_submission[list_classes] = y_test
    
    history = History.history
    history['auc'] = callback.aucs
    if kfold != 0:
        history['val_auc'] = callback.val_aucs
        
        sample_submission.to_csv(output_prefix+'_submission_fold'+str(ifold)+'.csv', index=False)
        import cPickle as pk
        with open(output_prefix+'_history'+str(ifold)+'.pkl', 'w') as f:
            pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)
    else:
        sample_submission.to_csv(output_prefix+'_submission_rerun.csv', index=False)
        import cPickle as pk
        with open(output_prefix+'_history_rerun.pkl', 'w') as f:
            pk.dump(history, f, protocol=pk.HIGHEST_PROTOCOL)
        
    print 'RNN Done!'