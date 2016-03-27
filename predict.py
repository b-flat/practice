# from sklearn.externals import joblib
import numpy as np
import requests
import sys
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

wordnet = WordNetLemmatizer()

def my_tokenize(doc):
   tok = word_tokenize(doc)
   return[wordnet.lemmatize(x) for x in tok]

    #Load vectorizer and model:
    count_vect = joblib.load('pickledModel/count_vect.pkl')
    tfidf_tranformer = joblib.load('pickledModel/tfidf_tranformer.pkl')
    model = joblib.load('pickledModel/logistic_model.pkl')

    #prepare data for prediction
    counts = count_vect.transform([text])
    article_tfidf = tfidf_tranformer.transform(counts)

    #predict and print
    classes =  model.classes_
    probas = model.predict_proba(article_tfidf)[0]
        

    #Prepare for printing:
    