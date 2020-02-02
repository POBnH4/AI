#!/usr/bin/env python
# coding: utf-8

# ## Import our libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
#for the sigmoid function we need expit() from scipy
import scipy.special
#library for plotting arrays
import matplotlib.pyplot as plt
# A particularly interesting backend, provided by IPython, is the inline backend. 
# This is available only for the Jupyter Notebook and the Jupyter QtConsole. 
# It can be invoked as follows: %matplotlib inline
# With this backend, the output of plotting commands is displayed inline 
# within frontends like the Jupyter notebook, directly below the code cell that produced it. 
# The resulting plots are inside this notebook, not an external window.
import random
import pandas as pd # to manage data frames and reading csv files
import glob
import imageio


# ## Set our Global Variables
# later you will need to modify these to present your solution to the Exercise

# In[2]:


#number of input, hidden and output nodes
input_nodes = 784 #we have a 28x28 matrix to describe each digit
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.6
batch_size = 1 # increase this if you want batch gradient descent

# epochs is the number of training iterations 
epochs = 10

# datasets to read
# you can change these when trying out other datasets
train_file = "mnist_train.csv"
test_file = "mnist_test.csv"


# ## Specify our Dataset for Classification
# Note we have indicated a train set for model training and test set for testing the learned model

# In[3]:


#load the mnist training data CSV file into a list
#train_data_file = open("mnist/mnist_train_100.csv", 'r') # open and read the 100 instances in the text file
train_data_file = open(train_file, 'r')
train_data_list = train_data_file.readlines() # read all lines into memory 
train_data_file.close() 
print("train set size: ", len(train_data_list))

#testing the network
#load the mnist test data CSV file into a list
test_data_file = open(test_file, 'r') # read the file 
test_data_list = test_data_file.readlines()
test_data_file.close()
print("test set size: ", len(test_data_list))


# In[4]:


class neuralNetwork:
    """Artificial Neural Network classifier.

    Parameters
    ------------
    lr : float
      Learning rate (between 0.0 and 1.0)
    ep : int
      Number of epochs
    bs : int
      Size of the training batch to be used when calculating the gradient descent. 
      batch_size = 1 standard gradient descent
      batch_size > 1 stochastic gradient descent 

    inodes : int
      Number of input nodes which is normally the number of features in an instance.
    hnodes : int
      Number of hidden nodes in the net.
    onodes : int
      Number of output nodes in the net.


    Attributes
    -----------
    wih : 2d-array
      Input2Hidden node weights after fitting 
    who : 2d-array
      Hidden2Output node weights after fitting 
    E : list
      Sum-of-squares error value in each epoch.
      
    Results : list
      Target and predicted class labels for the test data.
      
    Functions
    ---------
    activation_function : float (between 1 and -1)
        implments the sigmoid function which squashes the node input

    """

    def __init__(self, inputnodes=784, hiddennodes=200, outputnodes=10, learningrate=0.6, batch_size=1, epochs=10):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #two weight matrices, wih (input to hidden layer) and who (hidden layer to output)
        #a weight on link from node i to node j is w_ij
        
        
        #Draw random samples from a normal (Gaussian) distribution centered around 0.
        #numpy.random.normal(loc to centre gaussian=0.0, scale=1, size=dimensions of the array we want) 
        #scale is usually set to the standard deviation which is related to the number of incoming links i.e. 
        #1/sqrt(num of incoming inputs). we use pow to raise it to the power of -0.5.
        #We have set 0 as the centre of the guassian dist.
        # size is set to the dimensions of the number of hnodes, inodes and onodes for each weight matrix
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #set the learning rate
        self.lr = learningrate
        
        #set the batch size
        self.bs = batch_size
        
        #set the number of epochs
        self.ep = epochs
        
        #store errors at each epoch
        self.E= []
        
        #store results from testing the model
        #keep track of the network performance on each test instance
        self.results= []
        
        #define the activation function here
        #specify the sigmoid squashing function. Here expit() provides the sigmoid function.
        #lambda is a short cut function which is executed there and then with no def (i.e. like an anonymous function)
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    # function to help management of batching for gradient descent
    # size of the batch is controled by self,bs
    def batch_input(self, X, y): # (self, train_inputs, targets):
        """Yield consecutive batches of the specified size from the input list."""
        for i in range(0, len(X), self.bs):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + self.bs], y[i:i + self.bs])
    
    #train the neural net
    #note the first part is very similar to the query function because they both require the forward pass
    def train(self, train_inputs, targets_list):
    #def train(self, train_inputs):
        """Training the neural net. 
           This includes the forward pass ; error computation; 
           backprop of the error ; calculation of gradients and updating the weights.

            Parameters
            ----------
            train_inputs : {array-like}, shape = [n_instances, n_features]
            Training vectors, where n_instances is the number of training instances and
            n_features is the number of features.
            Note this contains all features including the class feature which is in first position
        
            Returns
            -------
            self : object
        """
      
        for e in range(self.ep):
            print("Training epoch#: ", e)
            sum_error = 0.0   
            for (batchX, batchY) in self.batch_input(train_inputs, targets_list):
                #creating variables to store the gradients   
                delta_who = 0
                delta_wih = 0
                
                # iterate through the inputs sent in
                for inputs, targets in zip(batchX, batchY):
                    #convert  inputs list to 2d array
                    inputs = np.array(inputs,  ndmin=2).T
                    targets = np.array(targets, ndmin=2).T

                    #calculate signals into hidden layer
                    hidden_inputs = np.dot(self.wih, inputs)
                    #calculate the signals emerging from the hidden layer
                    hidden_outputs = self.activation_function(hidden_inputs)

                    #calculate signals into final output layer
                    final_inputs=np.dot(self.who, hidden_outputs)
                    #calculate the signals emerging from final output layer
                    final_outputs = self.activation_function(final_inputs)
        
                    #to calculate the error we need to compute the element wise diff between target and actual
                    output_errors = targets - final_outputs
                    #Next distribute the error to the hidden layer such that hidden layer error
                    #is the output_errors, split by weights, recombined at hidden nodes
                    hidden_errors = np.dot(self.who.T, output_errors)
                                  
                    ## for each instance accumilate the gradients from each instance
                    ## delta_who are the gradients between hidden and output weights
                    ## delta_wih are the gradients between input and hidden weights
                    delta_who += np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
                    delta_wih += np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
                    
                    sum_error += np.dot(output_errors.T, output_errors)#this is the sum of squared error accumilated over each batced instance
                   
                pass #instance
            
                # update the weights by multiplying the gradient with the learning rate
                # note that the deltas are divided by batch size to obtain the average gradient according to the given batch
                # obviously if batch size = 1 then we simply end up dividing by 1 since each instance forms a singleton batch
                self.who += self.lr * (delta_who / self.bs)
                self.wih += self.lr * (delta_wih / self.bs)
            pass # batch
            self.E.append(np.asfarray(sum_error).flatten())
            print("errors (SSE): ", self.E[-1])
        pass # epoch
    
    #query the neural net
    def query(self, inputs_list):
        #convert inputs_list to a 2d array
        inputs = np.array(inputs_list, ndmin=2).T 
                
        #propogate input into hidden layer. This is the start of the forward pass
        hidden_inputs = np.dot(self.wih, inputs)
        
        #squash the content in the hidden node using the sigmoid function (value between 1, -1)
        hidden_outputs = self.activation_function(hidden_inputs)
                
        #propagate into output layer and the apply the squashing sigmoid function
        final_inputs = np.dot(self.who, hidden_outputs)
        
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
     
    #iterate through all the test data to calculate model accuracy
    def test(self, test_inputs, test_targets):
        self.results = []
        
        #go through each test instances
        for inputs, target in zip(test_inputs, test_targets):
            #query the network with test inputs
            #note this returns 10 output values ; of which the index of the highest value
            # is the networks predicted class label
            outputs = self.query(inputs)
            #get the target which has 0.99 as highest value corresponding to the actual class
            target_label = np.argmax(target)
            #get the index of the highest output node as this corresponds to the predicted class
            predict_label = np.argmax(outputs) #this is the class predicted by the ANN
        
            self.results.append([predict_label, target_label])
            pass
        pass
        self.results = np.asfarray(self.results) # flatten results to avoid nested arrays
    
        
    


# In[5]:


def get_my_test_data(folder):
    # our own image test data set
    X = []
    y = []
    
    # to read jpg change the regex to '/*.jpg'
    folder_expr = folder + '/*.png'
    print(folder_expr)

    for image_file_name in glob.glob(folder_expr): 
        print ("loading ... ", image_file_name)

        # load image data from png files into an array
        img_array = imageio.imread(image_file_name, as_gray=True)
        # reshape from 28x28 to list of 784 values, invert values
        img_data  = 255.0 - img_array.reshape(784)
        # then scale data to range from 0.01 to 1.0
        inputs = (img_data / 255.0 * 0.99) + 0.01
        
        # use the filename to set the correct label
        digit_class = int(image_file_name[-5:-4]) #negative indices for indexing from the end of the array
        
        X.insert(len(X), inputs)
        y.insert(len(y), digit_class)
       
        pass
    return(X,y)
pass


# In[6]:


def map_target_to_output_layer(instances, targets):
    X=[]
    Y=[]
    for inputs, target in zip(instances, targets):
        # create the target output values (all 0.01, except the desired label which is 0.99)
        y_vec = np.zeros(output_nodes) + 0.01
        y_vec[int(target)] = 0.99
        #print('output', target)
        
        X.insert(len(X), inputs) # simply inserting these they are already in the correct format
        Y.insert(len(Y), y_vec) # inserting these after the vector mapping
    pass
    return(X,Y)
pass

X_my_test, y_my_test = get_my_test_data('handwrittingnumbs')
X_my_test, y_my_test = map_target_to_output_layer(X_my_test, y_my_test)


# ## Helper functions to preprocess the data

# In[7]:


# preprocess the data directly read from the csv file
# we want to maintain inputs (X) and targets (y) seperatly
def preprocess_data(Xy):
    X=[]
    y=[]
    for instance in Xy:
        # split the record by the ',' commas
        all_values = instance.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        X.insert(len(X), inputs)
        #print(len(y), targets)
        y.insert(len(y), targets)
    pass
    return(X,y)
pass


# In[8]:


# numpy.random.choice generates a random sample from a given 1-D array
# we can use this to select a sample from our training data in case we want to work with a small sample
# for instance we use a small sample here such as 1500 instead of the larger 6000 instances
np.random.seed(4);
mini_training_data = np.random.choice(train_data_list, 500, replace = False)
print("Percentage of training data used:", (len(mini_training_data)/len(train_data_list)) * 100)

X_train, Y_train = preprocess_data(mini_training_data)


# In[9]:


# numpy.random.choice generates a random sample from a given 1-D array
# we can use this to select a sample from our training data in case we want to work with a small sample
# for instance we use a small sample here such as 1500
#mini_training_data = np.random.choice(train_data_list, 50, replace = False)
#print("Percentage of training data used:", (len(mini_training_data)/len(train_data_list)) * 100)
#X_train, Y_train = preprocess_data(mini_training_data)

print("This will take a few moments ...")
n_list = []
batch_sizes = [1, 5, 10, 15 , 25, len(mini_training_data)] #batch_size = batch_sizes[i]
epoch_size = [1, 5, 10, 15, 25, len(mini_training_data)] #epochs = epoch_size[i]
learning_rate = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] #learningrate = learning_rate[i]
hidden_nodes = [1, 5, 10, 50, 200, 400]
for i in range(len(epoch_size)):
    n = neuralNetwork(hiddennodes = hidden_nodes[i])
    n.train(X_train, Y_train)
    n_list.append(n) # maintain each ANN model in a list
    pass


# ## Test the ANN and compute the Accuracy
# We will keep track of the predicted and actual outputs in order to 
# calculate the accuracy of the model on the unseen test data. 
# 

# In[10]:


n.test(X_my_test, y_my_test)
#print network performance as an accuracy metric
correct = 0 # number of predictions that were correct

#iteratre through each tested instance and accumilate number of correct predictions
for result in n.results:
    if (result[0] == result[1]):
            correct += 1
    pass
pass

# print the accuracy on test set
print ("Test set accuracy% = ", (100 * correct / len(n.results)))


# # Gather the test results for each ANN

# In[11]:


#iteratre through each model and accumilate number of correct predictions
model_results = []
for model in n_list: 
    correct = 0
    model.test(X_my_test, y_my_test)
    for result in model.results:
        if (result[0] == result[1]):
                correct += 1
        pass
    correct = 100 * (correct/len(model.results))
    model_results.append(correct)
    print(correct)
    pass
pass


# print the accuracy on test set
#print ("Test set accuracy% = ", (100 * correct / len(n.results)))


# In[12]:


#objects = learningrate
#objects = epoch_size
objects = hidden_nodes
y_pos = np.arange(len(objects))
performance = model_results
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.xlabel('Hidden Nodes')
 
plt.show()


# In[ ]:




