#!/usr/bin/env python
# coding: utf-8

# ### Imports
# Let’s start off by importing the classes and functions required for this model and initializing the random number generator to a constant value to ensure we can easily reproduce the results.

# In[1]:


import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(1337)


# ### Load the Dataset
# We need to load the IMDB dataset. We are constraining the dataset to the top 5,000 words. We also split the dataset into train (50%) and test (50%) sets

# In[2]:


# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# modify the default parameters of np.load
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# restore np.load for future normal usage
np.load = np_load_old


# ### Pad the Input
# 
# Next, we need to truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in Keras

# In[3]:


# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# ### Neural Network Architecture
# 
# Now lets establish the architecture of our RNN.
# 
# The first layer is the Embedded layer that uses 32 length vectors to represent each word. The next layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a classification problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (positive review and negative review) in the problem.

# In[4]:


def create_model(top_words, max_review_length, embedding_size):
    model = Sequential()
    model.add(Embedding(top_words, embedding_size, input_length=max_review_length))
    #model.add(Dropout(0.2))
    model.add(LSTM(100))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


# ### Fit the Architecture
# 
# Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. The model is fit for only 3 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates.

# In[ ]:


embedding_size = 32

loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = 'accuracy'

epochs = 3
batch_size = 64


model = create_model(top_words, max_review_length, embedding_size)
model.compile(loss=loss, 
              optimizer=optimizer,
              metrics=[metrics])

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=epochs,
          batch_size=batch_size)


# ### Accuracy Measure 
# 
# Finally, we can evaluate the model on our test set to determine our accuracy.

# In[ ]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

