#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:32:26 2022

@author: pmane
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from flask import Flask,render_template,request
import pickle
import tensorflow_hub as hub
m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

import tokenization
import tensorflow as tf


from bert import bert_tokenization
BertTokenizer = bert_tokenization.FullTokenizer
app=Flask(__name__)

@app.route("/")
def fun1():
    return render_template("nlp1.html") # it provides html page as response

max_len=250
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocab_file, do_lower_case)

@app.route('/home')
def home():
    return render_template('nlp1.html')

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

@app.route("/predict",methods=["POST","GET"])
def fun2():
    Pclass=request.form["PClass"]
    encoder=LabelEncoder()
    encoder.classes_ = np.load('classes1.npy',allow_pickle=True)
    model = tf.keras.models.load_model('model3')
    x11=pd.DataFrame([np.array(Pclass)],columns=["Comments"])
    test_input = bert_encode(x11["Comments"], tokenizer, max_len=max_len)
    ypp=model.predict(test_input)
    ypp1=ypp.argmax(axis=1)
    if ypp.max()>=0.50:
        ypp2=encoder.inverse_transform(ypp1)
    else:
        ypp2=["Others"]
    return render_template('output.html', variety=ypp2[0])
    #return "Item Concern for entered user comment is {}".format(ypp2)
    

if __name__=="__main__":
    app.run(debug=True)
