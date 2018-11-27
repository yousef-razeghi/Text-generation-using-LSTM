#!/usr/bin/env python
# coding: utf-8

# In[48]:


#/*Yousef Razeghi S010725-Ozyegin University-Department of Computer Science*/
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import collections
from matplotlib import pyplot as plt
import datetime #clock training time
import time
import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# In[49]:


# ./LSTM-EPOCH-critique0-of-200-2018-05-23-21:53:07.897104
import os
os.getcwd()


# In[ ]:





# In[110]:


sess_load=tf.Session()
saver=tf.train.import_meta_graph("./LSTM-EPOCH-critique6-of-200-2018-05-23-23:28:30.869060.meta")
saver.restore(sess_load,"./LSTM-EPOCH-critique6-of-200-2018-05-23-23:28:30.869060")
graph_load=tf.get_default_graph()
#./LSTM-EPOCH-critique0-of-200-2018-05-23-23:06:30.647180


prediction=graph_load.get_tensor_by_name("add:0")
x=graph_load.get_tensor_by_name("x:0")

tf.reset_default_graph()


# In[51]:


data_file = {0:'./Edgard-Allan-Poe-Complete.txt',1: './Edgard-Allan-Poe-V-1.txt',
             2:'./Edgard-Allan-Poe-V-2.txt',3:'./Edgard-Allan-Poe-V-3.txt',
             4:'./Edgard-Allan-Poe-V-4.txt',5:'./Edgard-Allan-Poe-V-5.txt',
             6:'./Debian-GNU-Linux.txt',7:'./Hackers-Heroes-Of-Computer.txt',
             8:'./Critique-Of-Pure-Reason.txt'}


# In[52]:


num_layers=3
sequence_length=40
max_len = 40
step = 2
num_units = 512
learning_rate = 0.001
batch_size = 200
epoch = 200
scale_factor = 0.5


# In[53]:


def normalizeString(s):
    s=re.sub(r"([!?])",r"1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s


# In[54]:


#read the whole text file once and return in lower case
def read_data(file_name):
    text = open(file_name, 'r').read()
    text= normalizeString(text)
    return text.lower()


# In[55]:


#extract features of text file(e.g. uniquie word, quantity of unique characters
#, turning the unique characters into corresponding one-hot vectors of length(quantity_of_unique_characters)
#, extract the desired output which is the character following the selected sequence of characters,
#in our case each sequence consistes of 40 characters, indicated by max_len)
def featurize(text):
    
    vocabulary = list(set(text))
    vocabulary.sort()
    vocabulary_len = len(vocabulary)
    
    input_tokens = []
    output_token = []
    
    for i in range(0, len(text) - sequence_length, step):
        input_tokens.append(text[i:i+sequence_length])
        output_token.append(text[i+sequence_length])

    train_data = np.zeros((len(input_tokens), sequence_length, vocabulary_len))
    target_data = np.zeros((len(input_tokens), vocabulary_len))

    for i , each in enumerate(input_tokens):
        for j, _token in enumerate(each):
            train_data[i, j, vocabulary.index(_token)] = 1
        target_data[i, vocabulary.index(output_token[i])] = 1
    return train_data, target_data, vocabulary, vocabulary_len


# In[56]:


tx=read_data(data_file[8])#The book ---> Critiques of Pure Reason bi Immanuel Kant
#TO READ ALL BOOKS UNCOMMENT BELOW AND COMMENT ABOVE
#tx=''
#for i in data_file:
#    tx=tx+read_data(data_file[i])


# In[57]:


train_data, target_data, vocabulary, vocabulary_len=featurize(tx)


# In[111]:


while True:
    desired_keyword=input('Enter a keyword to generate the text:')
    match_cases=[m.start() for m in re.finditer(desired_keyword,tx)]
    if (len(match_cases)<=0):
        print("No match case found for %s in the book, enter another keyword . . ."%(desired_keyword))
    else:
        random_index=list(range(0,len(match_cases)-1))
        random.shuffle(random_index)
        random_ind=random_index.pop()
        random_sequence=tx[match_cases[random_ind]:((match_cases[random_ind])+(sequence_length))]
        random_phrase_colored=bcolors.BOLD+bcolors.FAIL+random_sequence[:len(desired_keyword)]+ bcolors.ENDC+random_sequence[len(desired_keyword):]
        print(random_sequence,'length of sequence',len(random_sequence))       
        print(random_phrase_colored)       
#        for i in range(len(match_cases)):
            #print('Found %s at %d'%(tx[match_cases[i]:match_cases[i]+len(desired_keyword)],match_cases[i]))
        break


# In[112]:


#this function takes a sequence of characters and turns
#that into corresponding one-hot vectors for each 
#character in the sequence, finally we can feed the output
#into the model and repeatedly generate characters
def seed_maker(input_tokens,vocabulary_len,vocabulary):
    test_seed = np.zeros((1,len(input_tokens), vocabulary_len)).astype('float32')

    for i , each in enumerate(list(input_tokens)):
        test_seed[:,i, vocabulary.index(each)] = 1
    return test_seed


# In[113]:


seed=seed_maker(random_sequence,vocabulary_len=vocabulary_len,vocabulary=vocabulary)


# In[115]:


seed_chars = ''
for each in seed[0]:
    seed_chars += vocabulary[np.where(each == max(each))[0][0]]
print ("Start seed chars: ", seed_chars)


# In[116]:


def sample(predicted):
    exp_predicted = np.exp(predicted/scale_factor)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities


# In[117]:


generated_id_sequence=[]

for i in range(5000):
    if i > 0:
        remove_fist_char = seed[:,1:,:]
        seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, vocabulary_len]), axis=1)
        for i in range(len(seed[0])):
            generated_id_sequence.append(np.argmax(seed[0][i]))
    
    predicted = sess_load.run((prediction), feed_dict = {x:seed})
    #print(predicted[:,1])
    
    predicted = np.asarray(predicted[0]).astype('float64')
    probabilities = sample(predicted)
    predicted_words = vocabulary[np.argmax(probabilities)]            
    seed_chars += predicted_words


# In[121]:


print('Number of unique words generated %d'%(len(list(set(seed_chars.split())))))


# In[119]:


seed_chars

