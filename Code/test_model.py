# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:08:17 2019

@author: abhishek
"""

from run_specs import *
from helper_nlp import *
from model_builder import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pandas.tools.plotting import table

######################Model Loading#########################################

from keras.models import load_model
model = load_model(wd + "/Model/" + saved_model_name +'.h5')

######################Data Processing#########################################

save_location = wd + "/Predictions/" 

#Import Test data
test_data = pd.read_csv(TEST_FILE, sep='\t')  

#Text Cleanup and strinsg and label generation
strings_test   = strings_list_test(  test_data, \
                                     reviewcol = review_column_name, \
                                     remove_stopwords=True, \
                                     stem_words=False)

#Tokenize words
with open(wd + "/Model/" + tokenizer_name + '.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Padding sequences
x_test = paddingsequence(tokenizer,MAX_SEQUENCE_LENGTH,strings_test)

#Prediction
predicted_sentiment = model.predict(x_test)
predicted_sentiment = np.where(predicted_sentiment > threshold,1,0).tolist()
predicted_sentiment = [item for sublist in predicted_sentiment for item in sublist]

#Submission
submission = pd.DataFrame({"id":test_data["id"].tolist(), \
                           "sentiment": predicted_sentiment})
    
submission.to_csv(save_location + predictions_file_name + ".csv", index = False)


##############################Testing on Sample Data############################

sample_data = pd.read_csv(SAMPLE_DATA_FILE)  

#Text Cleanup and strinsg and label generation
strings_test   = strings_list_test(  sample_data, \
                                     reviewcol = review_column_name, \
                                     remove_stopwords=True, \
                                     stem_words=False)

#Padding sequences
x_test = paddingsequence(tokenizer,MAX_SEQUENCE_LENGTH,strings_test)

#Prediction
predicted_sentiment_sample = model.predict(x_test).tolist()
predicted_sentiment_sample = [item for sublist in predicted_sentiment_sample for item in sublist]

#Sample PLot Table
sample_data["Prob rating >= 7"] = predicted_sentiment_sample
sample_data.to_csv(save_location +  "predictions_sample_data.csv", index = False)

plt.bar(sample_data["id"],sample_data["Prob rating >= 7"])
plt.title("Response on Arbitrary Reviews")
plt.savefig(save_location + "custom_reviews.png")


