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
np.random.seed(233)
os.environ["OMP_NUM_THREADS"] = "48"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, PReLU, add
from keras.layers import Reshape, Concatenate, Conv2D, MaxPool2D
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
embedding_path = path+"crawl-300d-2M.vec"
#embedding_path = path+"glove.840B.300d.txt"
embed_size = 300
max_features = 100000
max_len = 300

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
train["comment_text"].fillna("no comment")
test["comment_text"].fillna("no comment")

#with open(path+'clean_data.pkl', 'r') as f:
#    train, test = pk.load(f)

#X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.1)

with open(path+'val_flag_'+str(kfold)+'fold.pkl','r') as f:
    val_flag = pk.load(f)
ind_trn = np.where(~val_flag[ifold])[0].tolist()
np.random.shuffle(ind_trn)
X_train, X_valid, Y_train, Y_valid = train.iloc[ind_trn,:], train.iloc[val_flag[ifold],:], y[ind_trn], y[val_flag[ifold]]
#X_train, X_valid, Y_train, Y_valid = train.iloc[~val_flag[ifold],:], train.iloc[val_flag[ifold],:], y[~val_flag[ifold]], y[val_flag[ifold]]

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

#####################################
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

#####################################
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

def build_model():
    input1 = Input(shape=(max_len,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(6, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    History = model.fit(X_train, Y_train, batch_size=256, epochs=24, validation_data=(X_valid, Y_valid), verbose=1, callbacks=[callback, early_stop])
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
    