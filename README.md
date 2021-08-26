# Disaster-Response-Pipeline-Project

## Table of Contents
* [About Project]() 
* [Installation]()
* [Instructions to run the code]()
* [Licensing]()
## About Project
#### This is the project of Udacity's data scientist nanodegree program 
  Here I'm going to apply the  ETL & ML pipeline process on text message data to classify disaster messages.

## Installation

The code was built on Python 3.8,All libraries are avilable in Anaconda distribution of Python. 
python code :

  1- process_data.py
  
  This python file read the dataset, clean the data, and then store it in a SQL database.
      
  2- train_classifier.py
  
  Tis python code split  data into a training and test sets. Then, create a ML pipeline that uses NLTK and scikit-learn's Pipeline and GridSearchCV.

the dataset Data provided by Figure Eight.

  1- Message data : is CSV file of text messages.
  
  2- Categories : is CSV file fore categorize text messages in Message data for use in the ML pipeline
    

## Instructions to run the code

follow the following commands to set up your environments.

  1- Run ETL pipeline (process_data.py) in the terminal
  
      python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  2- Run ML pipeline (train_classifier.py)
  
      python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
      
  3- Run the following command in the app's directory to run deploy web app.
  
       python run.py
  

  4- Go to http://0.0.0.0:3001/ or localhost:3001
  
## Licensing
The dataset is provided by Figure Eight in Udacity data scientist nanodegree program.
