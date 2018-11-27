#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data_file = {0:'./Edgard-Allan-Poe-Complete.txt',1: './Edgard-Allan-Poe-V-1.txt',
             2:'./Edgard-Allan-Poe-V-2.txt',3:'./Edgard-Allan-Poe-V-3.txt',
             4:'./Edgard-Allan-Poe-V-4.txt',5:'./Edgard-Allan-Poe-V-5.txt',
             6:'./Debian-GNU-Linux.txt',7:'./Hackers-Heroes-Of-Computer.txt',
             8:'./Critique-Of-Pure-Reason.txt'}


# In[3]:


num_layers=3
sequence_length=40
max_len = 40
step = 2
num_units = 512
learning_rate = 0.001
batch_size = 200
epoch = 200
scale_factor = 0.5


# In[4]:


def normalizeString(s):
    s=re.sub(r"([!?])",r"1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s


# In[5]:


#read the whole text file once and return in lower case
def read_data(file_name):
    text = open(file_name, 'r').read()
    text= normalizeString(text)
    return text.lower()


# In[6]:


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


# In[7]:


tx=read_data(data_file[8])#The book ---> Critiques of Pure Reason bi Immanuel Kant
#TO READ ALL BOOKS UNCOMMENT BELOW AND COMMENT ABOVE
#tx=''
#for i in data_file:
#    tx=tx+read_data(data_file[i])


# In[8]:


train_data, target_data, vocabulary, vocabulary_len=featurize(tx)


# In[ ]:





# In[9]:


def sample(predicted):
    exp_predicted = np.exp(predicted/scale_factor)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities


# In[10]:


def get_a_cell(lstm_size, keep_prob):#LSTM cell generator to put in multilayer structure, considers dropout
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop


# In[11]:


def rnn(vocabulary_len, bias, weight, x):#Define Multi Layer LSTM and Get Prediction
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, vocabulary_len])
    x = tf.split(x, sequence_length, 0)
    keep_probability=1.0
        
    with tf.name_scope('lstm'):
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(num_units, keep_probability) for _ in range(num_layers)])

    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction


# In[ ]:





# In[12]:


g=tf.Graph()
with g.as_default():
    
    x = tf.placeholder(tf.float32, [None, sequence_length, vocabulary_len],name='x')
    y = tf.placeholder(tf.float32, [None, vocabulary_len],name='y')
    weight = tf.Variable(tf.random_normal([num_units, vocabulary_len]),name='weight')
    bias = tf.Variable(tf.random_normal([vocabulary_len]),name='bias')

    prediction = rnn( vocabulary_len, bias,weight, x)
    
    #print(prediction.name)
    
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    loss = tf.reduce_mean(softmax)
    
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    sess = tf.InteractiveSession(graph=g)
    tf.global_variables_initializer().run()


# In[14]:


saver=tf.train.Saver()


# In[15]:


#text = read_data(data_file[8])
#train_data, target_data, unique_words, len_unique_words = featurize(text)
total_loss=[]
epoch_mean_loss=[]
num_batches = int(len(train_data)/batch_size)

for i in range(epoch):
    print ("----------- Epoch {0}/{1} -each EPOCH-{2} Iterations--\n".format(i+1, epoch,num_batches))
    count = 0
    for _ in range(num_batches):
        train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
        count += batch_size
        _loss,_opt=sess.run([loss,optimizer] ,feed_dict={x:train_batch, y:target_batch})
        epoch_mean_loss.append(_loss)
        print('--Loss = %2.4f ' % (_loss),end='\r')
    print ("\n--------- Epoch {0}/{1} -finished at {2} --\n".format(i+1, epoch,str(datetime.datetime.now())))
    total_loss.append(np.mean(np.asarray(epoch_mean_loss)))
    epoch_mean_loss=[]
          
    path_epoch='./LSTM-EPOCH-critique'+str(i)+'-of-'+str(epoch)+'-'+str(datetime.datetime.now()).replace(" ","-")
    saver.save(sess, path_epoch)
    print('Session EPOCH %d of %d saved in \n %s\n'%(i,epoch,path_epoch))
path_full='./LSTM-FULL-critique-'+str(datetime.datetime.now()).replace(" ","-")
saver.save(sess, path_full)
print('Full session saved in \n %s\n'%(path_full))

#print("\ttensorboard --logdir=%s" % (logs_path))


# In[16]:


total_loss


# In[17]:


plt.plot(np.asarray(total_loss))


# In[18]:


sess_load=tf.Session()
saver=tf.train.import_meta_graph("./LSTM-EPOCH-critique6-of-200-2018-05-23-23:28:30.869060.meta")
saver.restore(sess_load,"./LSTM-EPOCH-critique6-of-200-2018-05-23-23:28:30.869060")
graph_load=tf.get_default_graph()



# In[19]:


tx2=read_data(data_file[0])#The book ---> Critiques of Pure Reason bi Immanuel Kant


# In[20]:


while True:
    desired_keyword=input('Enter a keyword to generate the text:')
    match_cases=[m.start() for m in re.finditer(desired_keyword,tx2)]
    if (len(match_cases)<=0):
        print("No match case found for %s in the book, enter another keyword . . ."%(desired_keyword))
    else:
        random_index=list(range(0,len(match_cases)-1))
        random.shuffle(random_index)
        random_ind=random_index.pop()
        random_sequence=tx2[match_cases[random_ind]:((match_cases[random_ind])+(sequence_length))]
        random_phrase_colored=bcolors.BOLD+bcolors.FAIL+random_sequence[:len(desired_keyword)]+ bcolors.ENDC+random_sequence[len(desired_keyword):]
        print(random_sequence,'length of sequence',len(random_sequence))       
        print(random_phrase_colored)       
#        for i in range(len(match_cases)):
            #print('Found %s at %d'%(tx[match_cases[i]:match_cases[i]+len(desired_keyword)],match_cases[i]))
        break


# In[21]:


#this function takes a sequence of characters and turns
#that into corresponding one-hot vectors for each 
#character in the sequence, finally we can feed the output
#into the model and repeatedly generate characters
def seed_maker(input_tokens,vocabulary_len,vocabulary):
    test_seed = np.zeros((1,len(input_tokens), vocabulary_len))

    for i , each in enumerate(list(input_tokens)):
        test_seed[:,i, vocabulary.index(each)] = 1
    return test_seed


# In[22]:


#train_batch=train_data[:20:]
#print('train_batch',train_batch.shape)
#seed = train_batch[:6:]
#print('seed',seed.shape)

seed=seed_maker(random_sequence,vocabulary_len=vocabulary_len,vocabulary=vocabulary)


# In[23]:


seed_chars = ''
for each in seed[0]:
    seed_chars += vocabulary[np.where(each == max(each))[0][0]]
print ("Seed: ", seed_chars)
test_seed=seed_chars
print('\n')


# In[24]:


generated_id_sequence=[]

for i in range(1000):
    if i > 0:
        remove_fist_char = seed[:,1:,:]
        seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, vocabulary_len]), axis=1)
        for i in range(len(seed[0])):
            generated_id_sequence.append(np.argmax(seed[0][i]))
    predicted = sess_load.run([prediction], feed_dict = {x:seed})
    predicted = np.asarray(predicted[0]).astype('float64')
    probabilities = sample(predicted[0])
    predicted_words = vocabulary[np.argmax(probabilities)]            
    seed_chars += predicted_words


# In[25]:


seed_chars


# In[ ]:





# In[ ]:




