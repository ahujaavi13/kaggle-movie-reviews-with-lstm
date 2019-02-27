# Code guide
### I have used `keras` on top of `tensorflow` to build this LSTM model. The role of each script is detailed out below

---

| Script           | Content                                                                           | To be changed when                                       |
|------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------|
| helper_nlp.py    | Contains list of functions that help prepare data for modeling                    | Need to update data preparation                          |
| model_builder.py | Contains complete LSTM model specifications,tokenizations and embeddings          | Need to update LSTM model structure                       |
| run_specs.py     | Contains locations to files and folder and very high level model   specifications | Need to update file locations and high level model specs |
| train_model.py   | Contains code to fit model with all specifications                                | Need to update model fitting strategey                   |
| test_model.py    | Contains code to predict for all test images                                      | Need to update test output/reports/performance summary   |

#### Other than the codes there are two log files as well which record progress of each run and gets updated for each run

| Log File      | Content                                                   |
|---------------|-----------------------------------------------------------|
| train_log.out | Contains details of model run when train_model.py was run |
| test_log.out  | Contains details of model run when test_model.py was run  |

### Steps to run the model:

* On **ubuntu** console ```ubuntu@ip-XX-XX-XX-XX:~$  source activate tensorflow_p36```  

  Doing this will automatically make tensorflow use **GPU**. If this activation is correctly 
  happening then the terminal will have something like this  

  ```
  (tensorflow_p36) ubuntu@ip-XX-XX-XX-XX:~$
  ```  


* **Clone**   
  ```
  git clone https://github.com/ahujaavi13/kaggle-movie-reviews-with-lstm.git
  ```  


* `cd` to folder `/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Code`  


* To **Train** model the following code snippet can be used
```(tensorflow_p36) ubuntu@ip-XX-XX-XX-XX:~/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Code$ nohup python ./train_model.py &> train_log.out &```  


* To see how training is progressing you can use tensorboard to visualize the training process on a dashboard which looks like this  

  ![picture](https://www.tensorflow.org/images/mnist_tensorboard.png)

  To do this you need to put the following code by suitably modifying `<your-user-name>`
  ```
  (tensorflow_p36) ubuntu@ip-XX-XX-XX-XX:~/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Code$ tensorboard --logdir /home/ubuntu/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Log/kaggle-movie-reviews-with-lstm
   ```  

   This will launch tensorboard on port `6006` (__May not be this port if multiple users are there using the server simultaneously__). Once tensorboard is launched it will be acessible from __your browser__ via the following link  

   `ip-address:6006`


* To **Test** model the following code snippet can be used
```(tensorflow_p36) ubuntu@ip-XX-XX-XX-XX:~/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Code$ nohup python ./test_model.py &> test_log.out &```  

  
* **Predictions** file will be available in `/home/ubuntu/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Predictions`  

* **Model** will be checkpointed and saved in `/home/ubuntu/users/<your-user-name>/kaggle-movie-reviews-with-lstm/Model`  

* **Tensorboard Log** will be saved in `/home/ubuntu/users/abhishek/kaggle-movie-reviews-with-lstm/Log`

* **Recommended Reading** --> [Sentiment Analysis](https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e)

---
