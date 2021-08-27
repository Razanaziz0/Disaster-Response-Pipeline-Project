import sys
import requests
import pandas as pd
import numpy as np
import datetime 
import sqlite3
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    
    
    messages = pd.read_csv(messages_filepath)
    messages.head()


    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    df = pd.merge(messages,categories,on=['id'])
    return df,categories
def clean_data(df,categories):



    # ### 3. Split `categories` into separate category columns.

    # cat_col find postion of categories column
    cat_col = categories.columns.get_loc('categories')
    # in df2 the code split categories into many columns
    df2 = categories['categories'].str.split(";", expand=True)
    # in this step the code drop categories column to add splitted categories
    categories=categories.drop(['categories'], axis = 1)

    # concat categories with ids 
    df3=pd.concat([categories.iloc[:, :cat_col], df2, categories.iloc[:, cat_col+1:]], axis=1)
    categories=df3



    # take the first row to make it as column lable for all categories column 
    category_colnames=categories.iloc[0]
    # remove '-' and digit to to unify it
    category_colnames=category_colnames.str.replace('-\d',"")

    # rename the columns of `categories`
    categories.columns =category_colnames
    categories.columns=('id',                'related',
                          'request',                  'offer',
                      'aid_related',           'medical_help',
                 'medical_products',      'search_and_rescue',
                         'security',               'military',
                      'child_alone',                  'water',
                             'food',                'shelter',
                         'clothing',                  'money',
                   'missing_people',               'refugees',
                            'death',              'other_aid',
           'infrastructure_related',              'transport',
                        'buildings',            'electricity',
                            'tools',              'hospitals',
                            'shops',            'aid_centers',
             'other_infrastructure',        'weather_related',
                           'floods',                  'storm',
                             'fire',             'earthquake',
                             'cold',          'other_weather',
                    'direct_report')
    categories.columns


    # ### 4. Convert category values to just numbers 0 or 1.

    for column in categories:
        # check if not id column and set each value to be the last character of the string
        if(categories[column] .dtype == object):

            categories[column] = categories[column].str.strip().str[-1]

            # convert column from string to numeric
            categories[column] =pd.to_numeric(categories[column])
    categories.head(100)


    # ### 5. Replace `categories` column in `df` with new category columns.

    # drop the original categories column from `df`

    df=df.drop(['categories'], axis = 1)

    df.head()


    df=pd.concat([df, categories], axis=1)
    # df1=pd.concat([df, categories], sort=False)
    df.head()


    # ### 6. Remove duplicates.


    # In[13]:


    # check number of duplicates
    df_dub=df.duplicated(subset='id').sum()
    df_dub
    #remove na values
    df_dropna = df.dropna(axis=0)

    # drop duplicates

    df=df_dropna.drop_duplicates(keep = 'first')


    # check number of duplicates
    df_dub=df.duplicated(subset='id').sum()
    df_dub
    return df



def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('categories', engine, index=False,if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':

    main()