import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt
import nltk
import string
import xgboost

from nltk.stem.porter import *
from sklearn import model_selection, preprocessing, linear_model, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def pattern_remover(text, pattern):
    st = re.findall(pattern, text)
    
    for i in st:
        text = re.sub(i, '', text)
    return text

def Preprocessing(data):
    data['new_tweets'] = np.vectorize(pattern_remover)(data['tweet'], "@[\w]*")
    data['new_tweets'] = data['new_tweets'].str.replace("[^a-zA-Z#]", " ")
    data['new_tweets'] = data['new_tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    tokens = data['new_tweets'].apply(lambda x : x.split())
    stemmer = PorterStemmer()
    tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])

    for i in range(len(tokens)):
        tokens[i] = ' '.join(tokens[i])
    return tokens

def feature_extraction(data):
    vect = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    tf = vect.fit_transform(data['new_tweets'])
    return tf

def accuracy(classifier, train_x, val_x, train_y, val_y):
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(val_x)
    
    acc = metrics.accuracy_score(predictions, val_y)
    return acc


data = train.append(test, ignore_index=True)
texts = Preprocessing(data)
data['new_tweets'] = texts
tfidf = feature_extraction(data)

train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]
train_x, val_x, train_y, val_y = model_selection.train_test_split(train_tfidf, 
                                                                  train['label'], 
                                                                  random_state=42, 
                                                                  test_size=0.3)

LR = accuracy(linear_model.LogisticRegression(), 
                            train_x, val_x, train_y, val_y)
print("Logistic Regression Accuracy : ",LR)

RF = accuracy(ensemble.RandomForestClassifier(), 
           train_x, val_x, train_y, val_y)
print("Random Forest Accuracy : ",RF)

SVM = accuracy(svm.SVC(), train_x, val_x, train_y, val_y)
print("SVM Accuracy : ",SVM)

xgb = model(xgboost.XGBClassifier(), train_x, val_x, train_y, val_y)
print("XGBoost Accuracy : ",xgb)
