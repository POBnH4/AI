#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

#for nlp
import nltk
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#text vectorisation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

#metrics
from sklearn.metrics import classification_report, accuracy_score

#import method releated to evaluation
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#for graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


filename = 'SMSSpamData.csv'
df = pd.read_csv(filename) 


# In[3]:


class_mapping = {label:idx for idx,label in enumerate(np.unique(df['class']))}

print(class_mapping)
class_labels = [x for x in class_mapping] # store the class labels for later


# In[4]:


#use the mapping dictionary to transform the class labels into integers

df["class"] = df["class"].map(class_mapping)


# In[5]:


df.head(3)


# In[6]:


df.loc[15, 'sms_msg']


# In[7]:


#import regular expressions to clean up the text
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # remove all html markup
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) # findall the emoticons
    
    # remove the non-word chars '[\W]+'
    # append the emoticons to end 
    #convert all to lowercase
    # remove nose char for consistency
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', '')) 
    return text


# In[8]:


preprocessor(df.loc[15, 'sms_msg'])


# In[9]:


# apply the preprocessor to the entire dataframe (i.e. column review)
df['sms_msg'] = df['sms_msg'].apply(preprocessor)


# In[10]:


# download the stopwords if not done before (need an Internet connection)
nltk.download('stopwords')


# A basic text preprocessing pipeline
# The basic pipeline includes stopword removal, tokenising and stemming. 

# In[11]:



stop = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def tokenizer(text):
       return text.split()

def tokenizer_stemmer(text):
    return [stemmer.stem(word) for word in tokenizer(text)]#text.split()]


def stop_removal(text):
       return [w for w in text if not w in stop]


# In[12]:


df.loc[180, 'sms_msg']


# #  Vectorisation of text data

# In[13]:


from sklearn.model_selection import train_test_split

X = df.loc[:, 'sms_msg'].values
y = df.loc[:, 'class'].values

text_train, text_test, y_train, y_test = train_test_split(X, y, 
                                                          random_state=42,
                                                          test_size=0.25,
                                                          stratify=y)


# # CountVectorizer

# In[14]:


vectorizer = CountVectorizer()
vectorizer.fit(X) # Learn a vocabulary dictionary of all tokens in the raw documents.

#X_train = vectorizer.transform(text_train)
#X_test = vectorizer.transform(text_test)

X_train = df.loc[:2500, 'sms_msg'].values
y_train = df.loc[:2500, 'class'].values
X_test = df.loc[2500:, 'sms_msg'].values
y_test = df.loc[2500:, 'class'].values

param_grid0 = [{'vect__ngram_range': [(1, 5)], #can also extract 2-grams of words in addition to the 1-grams (individual words)
               'vect__stop_words': [stop, None], # use the stop dictionary of stopwords or not
               'vect__tokenizer': [tokenizer_stemmer]}, # use a tokeniser and the stemmer 
               ]


# In[15]:


print(X_train.shape)
print(X_test.shape)


# In[16]:


print('Sice of the vocabulary or the number of features in the vector ', len(vectorizer.vocabulary_))


# In[17]:


print(vectorizer.get_feature_names()[2000:2020]) #Array mapping from feature integer indices to feature name


# In[18]:


from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

import nltk
from nltk.stem.porter import PorterStemmer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


mnb_tfidf = Pipeline([('vect', tfidf),
                     ('clf',  MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])


                   
gs_mnb_tfidf = GridSearchCV(mnb_tfidf, param_grid0,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=1) 
gs_mnb_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_mnb_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_mnb_tfidf.best_score_)
clf = gs_mnb_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[ ]:




