# **Movie reviews with LSTM**

In this repository I have tried to perform `sentiment analysis` using imdb movie reviews data available in Kaggle. Download it from **[here](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)**.
While doing that I have also leveraged pre-trained word embeddings by google which is an example of `transfer learning`. For this I have used Google's word2vec embedding.
Read about it more from **[here](https://code.google.com/archive/p/word2vec/)** and download it from **[here](https://www.kaggle.com/ymtoo86/googlenews-vectors-negative300)**.

## **Folder Guide**
```
|-----kaggle-movie-reviews-with-lstm  
		|----Code                                     #Codes for running Model			                      
			|----helper_nlp.py
			|----model_builder.py
			|----run_specs.py
			|----train_model.py
			|----test_model.py
			|----train_log.out
			|----test_log.out
		|----Log                                      #For Logging Tensorboard Output
		|----Model			                          #For saving model  
		|----Predictions                              #For Saving Predictions
		|----Sample_Data                              #Sample Data used for Modeling 
```
## Data

I have tried to predict the probability of a review getting a rating of more than 7. The complete dataset
has been downloaded from `Kaggle` and the inspiration is drawn from a competition which can be viewed **[here](https://www.kaggle.com/c/word2vec-nlp-tutorial)**. The
code currently generates submission file which can submitted to the competition to benchmark its accuracy. The current accuracy is slightly over .8 (scope of improvement)

<img src="https://raw.githubusercontent.com/ahujaavi13/kaggle-movie-reviews-with-lstm/master/Predictions/kaggle_accuracy.png" width=800 height = 400>

Once the algorithm is ready and tuned properly it will do sentiment classification as it has been illustrated below from a dummy review data that has been created and kept in
`Sample_Data`. Input the reviews of your own. The predictions on my reviews are coming as follows  

| id           | review                                                                                                                                                                                                | Prob rating >= 7 |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| MI 6         | A lovely evening spent watching tom cruise in mission impossible 6.   Totally worth the time                                                                                                          | 1                |
| Stree        | Stree started off not so terribly but had one of the worst endings   although Rajkumar Rao was fantastic                                                                                              | 0.000176944      |
| Dangal       | watching amir khan in dangaal has been an absolute delight. One of the   best movies of recent times                                                                                                  | 0.95155108       |
| Sacred Games | Although very interesting and thrilling from the start it seemed to be a   stretch after a while with predictable twists.The acting and cinematography   is brilliant but plot could have been better | 0.223794714      |

The distribution of the probabilities are as follows which seem to align with the nature of the reviews

<img src="https://raw.githubusercontent.com/ahujaavi13/kaggle-movie-reviews-with-lstm/master/Predictions/custom_reviews.png" width=600 height = 400>

The ROC curve for the current model is as follows

<img src="https://raw.githubusercontent.com/ahujaavi13/kaggle-movie-reviews-with-lstm/master/Predictions/roc_on_validation.png" width=600 height = 400>
