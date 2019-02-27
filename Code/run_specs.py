# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:08:17 2019

@author: abhishek
"""
import subprocess

#The following code automatically detects root directory
wd  = subprocess.check_output('git rev-parse --show-toplevel', \
                               shell=True).decode('utf-8').strip()

#If you want to delete all previously stored logs and models
clear_log = "yes"
clear_model = "yes"


##########################Data Specifications#####################

review_column_name    = "review"
sentiment_column_name = "sentiment"
id_column_name        = "id"

#######################Model Specifications######################

transfer_learning_repo  = '/home/ubuntu/transfer_learning_repo/pretrained-embedding-vector-by-google'
datasource              = "/home/ubuntu/Data_Warehouse/movie-reviews-data"

EMBEDDING_FILE        = transfer_learning_repo + '/GoogleNews-vectors-negative300.bin'
TRAIN_FILE            = datasource + '/labeledTrainData.tsv'
TEST_FILE             = datasource + '/testData.tsv'
SAMPLE_DATA_FILE      = wd + "/Sample_Data/testing_reviews_model.csv"

MAX_SEQUENCE_LENGTH   = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
MODEL_NAME        = "kaggle-movie-reviews-with-lstm"

#####################Training Specifications#######################

epochs = 35
batch_size= 64

#################Model Saving####################################

tokenizer_name   = "tokenizer"
saved_model_name = "movie-review-sentiment-prediction"
validation_plot_name = "roc_on_validation"
#################Predictions generation##########################

threshold = .5
predictions_file_name = "submission"

