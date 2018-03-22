#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:08:11 2018

@author: ldong
"""

import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('/org/centers/ccgo/ldong/twitter_bigrams.bin')
emb = model.embed_sentence("once upon a time .") 
embs = model.embed_sentences(["first sentence .", "another sentence"]) 

import cPickle as pk
with open('data/clean_data.pkl', 'r') as f:
    train, test = pk.load(f)

list_sentences_train = train["comment_text"].fillna("_na_").values.tolist()
list_sentences_train = [s.decode('utf8') for s in list_sentences_train]
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values.tolist()
list_sentences_test = [s.decode('utf8') for s in list_sentences_test]

embed_trn = model.embed_sentences(list_sentences_train)
embed_tst = model.embed_sentences(list_sentences_test)

with open('data/embed_sent2vec.pkl', 'w') as f:
    pk.dump([embed_trn, embed_tst], f, protocol=pk.HIGHEST_PROTOCOL)