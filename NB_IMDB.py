#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

import nltk
from nltk.stem.porter import PorterStemmer


# In[2]:


count = CountVectorizer() #from sklearn.feature_extraction.text import CountVectorizer
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)


# In[3]:


print(count.vocabulary_) # vocabulary_ attribute of CountVectorizer() shows a mapping of terms to feature indices.


# In[4]:


print(bag.toarray())


# In[5]:


count_2 = CountVectorizer(ngram_range=(1,2))
bag_2 = count_2.fit_transform(docs)
print(count_2.vocabulary_)
print(bag_2.toarray())


# In[6]:


np.set_printoptions(precision=2) # These options determine the way floating point numbers are displayed.


# In[7]:


tfidf = TfidfTransformer(use_idf=True, 
                         norm='l2', 
                         smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs))
      .toarray())


# In[8]:


tf_is = 3 # suppose term "is" has a frequency of 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)


# In[9]:


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf 


# In[10]:


l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf


# In[11]:


corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.shape)


# In[12]:


vectorizer_123 = TfidfVectorizer(ngram_range=(1,3))
X_123 = vectorizer_123.fit_transform(corpus)
print(vectorizer_123.get_feature_names())

print(X_123.shape)


# In[13]:


vectorizer_mm = TfidfVectorizer(max_df=1.0,min_df=0.5)
X_mm = vectorizer_mm.fit_transform(corpus)
print(vectorizer_mm.get_feature_names())

print(X_mm.shape)


# In[14]:


df = pd.read_csv('movie_data_cat.csv', encoding='utf-8')
df.head(10)


# In[15]:


df.shape
df.columns


# In[16]:


class_mapping = {label:idx for idx,label in enumerate(np.unique(df['sentiment']))}

print(class_mapping)

#use the mapping dictionary to transform the class labels into integers

df['sentiment'] = df['sentiment'].map(class_mapping)
df.head(10)


# In[17]:


df.loc[5635, 'review']#[-50:]


# In[18]:


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


# In[19]:


preprocessor(df.loc[3635, 'review'])#[-50:]


# ## Apply the clean data preprocessor to the text

# In[20]:


preprocessor("</a>This :) is :( a test :-)!")


# In[21]:


# apply the preprocessor to the entire dataframe (i.e. column review)
df['review'] = df['review'].apply(preprocessor)


# ## Tokenise - break text into tokens

# In[22]:


def tokenizer(text):
       return text.split()


# In[23]:


print(tokenizer("Tokenise this sentence into its individual words"))


# In[24]:


from nltk.corpus import stopwords 

nltk.download('stopwords')


# create a method to accept a piece of tokenised text and return text back without the stopped words

# In[25]:


stop = set(stopwords.words('english'))
def stop_removal(text):
       return [w for w in text if not w in stop]


# In[26]:


text = "This is a sample sentence, demonstrating the removal of stop words."
stopped_text = stop_removal(text.split())
print(stopped_text) 


# ## Stemming - Processing tokens into their root form

# In[27]:


from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#See which languages are supported.
print(" ".join(SnowballStemmer.languages))


# In[28]:


#get the english stemmer
stemmer = SnowballStemmer("english")

#stem a word
print(stemmer.stem("running"))


# In[29]:


#Decide not to stem stopwords with ignore_stopwords
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)

#compare the two versions of the stemmer
print(stemmer.stem("having"))

print(stemmer2.stem("having"))


# In[30]:


#The 'english' stemmer is better than the original 'porter' stemmer.
print(SnowballStemmer("english").stem("generously"))

print(SnowballStemmer("porter").stem("generously"))


# # Tokenise + Stemming 

# In[31]:


def tokenizer_stemmer(text):
    return [stemmer.stem(word) for word in tokenizer(text)]#text.split()]


# In[32]:


tokenizer('runners like running and thus they run')


# In[33]:


tokenizer_stemmer('runners like running and thus they run')


# You can clearly see from the code above the effect of the stemmer on the tokens

# In[34]:


from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_stemmer('A runner likes running and runs a lot')[-8:]
if w.lower() not in stop]


# # Training a model for sentiment classification

# In[35]:


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

### smaller sample
X_train = df.loc[:2500, 'review'].values
y_train = df.loc[:2500, 'sentiment'].values


# In[36]:


param_grid0 = [{'vect__ngram_range': [(1, 5)], #can also extract 2-grams of words in addition to the 1-grams (individual words)
               'vect__stop_words': [stop, None], # use the stop dictionary of stopwords or not
               'vect__tokenizer': [tokenizer_stemmer]}, # use a tokeniser and the stemmer 
               ]


# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV



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


# In[38]:


gs_mnb_tfidf.fit(X_train, y_train)


# In[39]:


print('Best parameter set: %s ' % gs_mnb_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_mnb_tfidf.best_score_)


# In[40]:


clf = gs_mnb_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[41]:


param_grid = [{'vect__ngram_range': [(1, 1)], #can also extract 2-grams of words in addition to the 1-grams (individual words)
               'vect__stop_words': [stop, None], # use the stop dictionary of stopwords or not
               'vect__tokenizer': [tokenizer]}, # use a tokeniser and the stemmer 
               ]

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


mnb_tfidf = Pipeline([('vect', tfidf),
                     ('clf',  MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])


                   
gs_mnb_tfidf = GridSearchCV(mnb_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=1) 
gs_mnb_tfidf.fit(X_train, y_train)
print('Best parameter set: %s ' % gs_mnb_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_mnb_tfidf.best_score_)
clf = gs_mnb_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[42]:


param_grid = [{'vect__ngram_range': [(1, 2)], #can also extract 2-grams of words in addition to the 1-grams (individual words)
               'vect__stop_words': [stop, None], # use the stop dictionary of stopwords or not
               'vect__tokenizer': [tokenizer_stemmer]}, # use a tokeniser and the stemmer 
               ]

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


mnb_tfidf = Pipeline([('vect', tfidf),
                     ('clf',  MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])


                   
gs_mnb_tfidf = GridSearchCV(mnb_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=1) 
gs_mnb_tfidf.fit(X_train, y_train)
print('Best parameter set: %s ' % gs_mnb_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_mnb_tfidf.best_score_)
clf = gs_mnb_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[43]:


from sklearn.neural_network import MLPClassifier

param_grid = [{'vect__ngram_range': [(1, 2)], #can also extract 2-grams of words in addition to the 1-grams (individual words)
               'vect__stop_words': [stop], # use the stop dictionary of stopwords or not
               'vect__tokenizer': [tokenizer_stemmer]}, # use a tokeniser and the stemmer 
               ]

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


mnb_tfidf = Pipeline([('vect', tfidf),
                     ('clf',  MLPClassifier(max_iter=100))])

                   
gs_mnb_tfidf = GridSearchCV(mnb_tfidf, param_grid,
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




