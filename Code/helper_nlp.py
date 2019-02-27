# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:08:17 2019

@author: abhishek
"""

import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", "", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def strings_labels_list_train(data,labelcol,reviewcol,remove_stopwords,stem_words):
    strings =[]
    labels = []
    for i in range(len(data)):
        string = data[reviewcol][i]
        label  = data[labelcol][i]
        strings.append(text_to_wordlist(string, \
                                        remove_stopwords=remove_stopwords, \
                                        stem_words=stem_words))
        labels.append(label)
    labels = np.array(labels)
    print('Found %s texts in Training Data' % len(strings))
    return strings,labels

def strings_list_test(data,reviewcol,remove_stopwords,stem_words):
    strings =[]
    for i in range(len(data)):
        string = data[reviewcol][i]
        strings.append(text_to_wordlist(string, \
                                        remove_stopwords=remove_stopwords, \
                                        stem_words=stem_words))
    print('Found %s texts in Test Data' % len(strings))
    return strings





def train_validation_create(strings,labels,VALIDATION_SPLIT):
    perm = np.random.permutation(len(strings))
    
    idx_train = perm[:int(len(strings)*(1-VALIDATION_SPLIT))]
    idx_val = perm[int(len(strings)*(1-VALIDATION_SPLIT)):]
    
    strings_train = strings[idx_train]
    labels_train = labels[idx_train]
    
    strings_val = strings[idx_val]
    labels_val = labels[idx_val]
    
    return strings_train,labels_train,strings_val,labels_val






