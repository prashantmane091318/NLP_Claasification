# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
import spacy
nlp = spacy.load("en_core_web_sm")
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from flask import Flask,render_template,request
import pickle
app=Flask(__name__)

@app.route("/")
def fun1():
    return render_template("nlp1.html") # it provides html page as response

def preprocess(msg):
    voc_size=1000
    review=re.sub('[^a-zA-Z]'," ",msg)
  #doc = nlp(review)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stopwords.words('english')]
    review=" ".join(review)
    doc = nlp(review)
    review=[token.lemma_ for token in doc]
    review=" ".join(review)
    one_hot_rep1=[one_hot(review,voc_size)]
    emb_doc1=pad_sequences(one_hot_rep1,maxlen=27,padding='pre')
    return emb_doc1

@app.route("/predict",methods=["POST","GET"])
def fun2():
    Pclass=request.form["PClass"]
    encoder=LabelEncoder()
    encoder.classes_ = np.load('classes.npy',allow_pickle=True)
    model = tf.keras.models.load_model('my_model')
    test=preprocess(Pclass)
    y_p=encoder.inverse_transform(model.predict_classes(test))
   
    return "Item Concern for entered user comment is {}".format(y_p)
    

if __name__=="__main__":
    app.run(debug=True)