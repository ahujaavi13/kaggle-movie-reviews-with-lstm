# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:08:17 2019

@author: abhishek
"""
import pandas as pd
from run_specs import * 
from helper_nlp import *
from model_builder import *
from keras.callbacks import TensorBoard as tb
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import os


#######################Remove old Logs and Models#######################

if clear_log == "yes":
	folder = wd +  "/Log/" + MODEL_NAME
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(str(e) + " does not exist")
			
if clear_model == "yes":
	folder = wd +  "/Model/" 
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(str(e) + " does not exist")

######################Data Processing#########################################

#Import training data
train_data = pd.read_csv(TRAIN_FILE, sep='\t')
train_data.head(5) 

#Text Cleanup and strinsg and label generation
strings,labels = strings_labels_list_train(train_data, \
                                         labelcol = sentiment_column_name, \
                                         reviewcol = review_column_name, \
                                         remove_stopwords=True, \
                                         stem_words=False)

#Tokenize words
tokenizer,word_index = tokenizedata(strings,MAX_NB_WORDS)

# saving
with open(wd + "/Model/" + tokenizer_name + '.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Padding sequences
sequence_data = paddingsequence(tokenizer,MAX_SEQUENCE_LENGTH,strings)

#Train and Validation Creation
x_train,y_train,x_val,y_val = train_validation_create(sequence_data,labels,VALIDATION_SPLIT)



#####################Model Fitting##########################################

save_location = wd + "/Predictions/" 

#Use pre-trained model to generate embedding
embedding_matrix = embeddingenerator(EMBEDDING_FILE,EMBEDDING_DIM,MAX_NB_WORDS,word_index)

#Initiating Tensorboard
tbCallBack = tb(log_dir= wd +  "/Log/" + MODEL_NAME,histogram_freq=0, batch_size=32, write_graph=True, write_grads= True, write_images=True)

model = kerasmodelbuilder(EMBEDDING_DIM,MAX_SEQUENCE_LENGTH, \
                      word_index,MAX_NB_WORDS,embedding_matrix)

model.fit(x_train, y_train,validation_data=(x_val, y_val), \
          epochs=epochs, \
          batch_size=batch_size, \
          callbacks = [tbCallBack]
          )


y_val_proba = model.predict(x_val)
fpr, tpr, _ = metrics.roc_curve(y_val, y_val_proba)
auc = metrics.roc_auc_score(y_val, y_val_proba)
plt.plot(fpr,tpr,label="validation data, auc="+str(auc))
plt.legend(loc=4)
plt.title("ROC on Validation")
plt.savefig(save_location + validation_plot_name + ".png")

#Fitting model on full Data

model.save(wd + "/Model/" + saved_model_name + ".h5")