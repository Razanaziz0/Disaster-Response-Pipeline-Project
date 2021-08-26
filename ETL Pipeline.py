
# # ETL Pipeline Preparation
# ### 1. Import libraries and load datasets.
# - Import Python libraries


# import libraries
import requests
import pandas as pd
import numpy as np
import datetime 


# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()


# load categories dataset
categories = pd.read_csv('categories.csv')
categories.head()


# ### 2. Merge datasets.


# merge datasets
df = pd.merge(messages,categories,on=['id'])
df.head()


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



# ### 7. Save the clean dataset into an sqlite database.

import sqlite3
from sqlalchemy import create_engine

engine = create_engine('sqlite:///categories.db')
df.to_sql('categories', engine, index=False)


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:




