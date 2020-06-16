# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:04:24 2017

"""
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
#
#EXAMPLE_TEXT = "This is a sample sentence, showing off the stop words filtration."
#
#stop_words = set(stopwords.words('english'))
#
#word_tocken = word_tokenize(EXAMPLE_TEXT)
#
#filtering_word = []
#
#for i in word_tocken:
#    if i not in stop_words:
#        filtering_word.append(i)
# 
#print(word_tocken)
#print(filtering_word)
#---------------------------------------------------------------------------------

#___________________partOfSpeech_________________________________________#
#import nltk
#from nltk.corpus import state_union
#from nltk.tokenize import PunktSentenceTokenizer
#

#
#custom = PunktSentenceTokenizer(Train_Data)
#  
#Tok = custom.tokenize(Sample_Data)
#def process():
#    try:
#      for i in Tok[:5]:
#         words =nltk.word_tokenize(i)
#         tagged = nltk.pos_tag(words)
#         print(tagged)
#    except Exception as e:
#        print(str(e))
#
#
#process()
#-----------------------------------------------------------------------------------
import nltk
import random
from nltk.corpus import movie_reviews
import Tkinter as tk
import numpy as np
import tkMessageBox as ms
from functools import partial
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print(documents[1])
#df = pd.DataFrame(documents[1:5], columns = ['Words', 'category'])
#print(df)
stop_words = set(stopwords.words('english'))

all_words = []
for w in movie_reviews.words():
    if w not in stop_words:
     all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["stupid"])


word_features = list(all_words.keys())[:3000]
#word_fValues = list(all_words.freq())[1:5]
#plt.plot(word_features[1:5])
#plt.boxplot(all_words.most_common(5))
#plt.hist(word_features)
plt.show()


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]
# set that we'll train our classifier with
training_set = featuresets[:1900]
#print(training_set)
# set that we'll test against.
testing_set = featuresets[1900:]
#print(testing_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15) 
 
#print( classifier.classify(find_features(movie_reviews.words('neg/cv000_29416.txt'))))






        
       

