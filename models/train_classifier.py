import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report,recall_score, make_scorer


def load_data(database_filepath):
    
    """load_data is function to load data from sqlite database
        
        Input:
            database_filepath --> the file location
            
        Output:
            X --> message data 
            Y --> 36 categories data
            category_names: categories columns names
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM categories", engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names=Y.columns
    
    return X, Y,category_names



url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    #tokenization function to process your text data

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def display_results(y_test, y_pred):
    
    labels = np.unique(y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
     
def build_model():
    """build_model function to specify pipeline and define parameters for GridSearchCV
        
    Output:
        GridSearchCV --> model for train and prediction 
    """    
    
    
    pipeline = Pipeline([('vect',   CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),
                     ('MLC', MultiOutputClassifier(KNeighborsClassifier()))])
    
    parameters = {'MLC__estimator__n_neighbors': [3,5],'MLC__estimator__leaf_size':[10,20,30] }
    custom_recall = make_scorer(recall_score,average='weighted')

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, verbose=2)


    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate_model is function to evaluate test data
    
    Input:
        model --> model was created by using build model
        X_test 
        Y_test
        category_names --> include 36 category names
        

    """    
    
    Y_pred = model.predict(X_test)
    display_results(Y_test, Y_pred)


def save_model(model, model_filepath):
    """ save_model is function save trained model to pickle file
        
        Input :
            model--> model after trained
            model_filepath --> string path to save pickle file
            
        Output:
            pickle file saved to specified path 'model_filepath'
    """  
    
    model_file = open(model_filepath,"wb")
    pickle.dump(model, model_file)
    model_file.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()