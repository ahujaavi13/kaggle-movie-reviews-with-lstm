# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:08:17 2019

@author: abhishek
"""
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation
from keras.models import Sequential

#Function for tokenization

def tokenizedata(strings,MAX_NB_WORDS):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(strings)
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))
    return tokenizer,word_index

#Function for padding input sequence to same size
def paddingsequence(tokenizer,MAX_SEQUENCE_LENGTH,strings):
    sequences = tokenizer.texts_to_sequences(strings)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data

#Function for generating embedding matrix
def embeddingenerator(EMBEDDING_FILE,EMBEDDING_DIM,MAX_NB_WORDS,word_index):
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
            binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))
    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix

#Function to create model
def kerasmodelbuilder(EMBEDDING_DIM,MAX_SEQUENCE_LENGTH, \
                      word_index,MAX_NB_WORDS,embedding_matrix):
    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    model = Sequential() 
    model.add(Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    print(model.summary()) 
    return model
