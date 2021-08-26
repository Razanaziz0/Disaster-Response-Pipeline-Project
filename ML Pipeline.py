
# import libraries
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
    # load data from database
    
    engine = create_engine(database_filepath)
    df = pd.read_sql("SELECT * FROM categories", engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    
    return X, Y


# ### 2. Write a tokenization function to process your text data

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
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


# ### 3. Build a machine learning pipeline
def build_model():

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),
                     ('MLC', MultiOutputClassifier(KNeighborsClassifier()))])
    return pipeline

def improve_model(model, X_test, Y_test):
#Finds best params for model via GridSearchCV
    model.get_params()
    parameters = {'MLC__estimator__n_neighbors': [3,5],'MLC__estimator__leaf_size':[10,20,30] }
    custom_recall = make_scorer(recall_score,average='weighted')
    cv = GridSearchCV(model, param_grid = parameters, n_jobs = -1, verbose=2)
    # cv.fit(X_train,Y_train)

    return cv


def evaluate_model(model, X_test, Y_test):
    
    Y_pred = model.predict(X_test)
    display_results(Y_test, Y_pred)

    
def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    
def save_model(model, model_filepath):

#Save model to pickle file.

#input pipeline model
#outputpickle file
      
    model_file = open(model_filepath,"wb")
    pickle.dump(model, model_file)
    model_file.close()


def main():
    X, Y = load_data('sqlite:///categories.db')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=42)
    
    # building model
    pipeline=build_model()
    # Training model
    pipeline.fit(X_train, y_train)
    cv.fit(X_train,y_train)
    # Evaluate model
    evaluate_model(pipeline, X_test, Y_test)
    # save model
    model_filename = 'model_MultiOutputClassifier.pkl'
    save_model(model, model_filename)

main()
# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[29]:


# pipeline.get_params()
# cv.best_params_



# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[22]:


