#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm_notebook as tqdm


# In[87]:


def spam(path):
    data = []
    label = []
    with open(path, 'r', encoding='ISO-8859-1') as read_file:
        for sent in tqdm(read_file.readlines()):
            sent = sent.strip()
            i = sent.find(',', 0, len(sent))
            label.append(sent[:i])
            data.append(sent[i+1:-3])
    return data[1:],label[1:]


# In[100]:


spam_data, spam_label = spam('../spam1.csv')


# In[101]:


len(spam_data)


# In[105]:


def rt(path):
    data = []
    label = []
    with open(path + 'rt-polarity.neg', 'r', encoding='ISO-8859-1') as read_file:
        for sent in tqdm(read_file.readlines()):
            data.append(sent.strip())
            label.append('neg')
            
    with open(path + 'rt-polarity.pos', 'r', encoding='ISO-8859-1') as read_file:
        for sent in tqdm(read_file.readlines()):
            data.append(sent.strip())
            label.append('pos')
    return data, label


# In[106]:


rt_data, rt_label = rt('../rt-polaritydata/rt-polaritydata/')


# In[103]:


len(rt_data)


# In[198]:


# 1st list data，2rd list label
# using tfidf to get 50-60 most frequent words
def handle(data, label):
    train_60_data = []
    train_50_data = []
    val_60_data = []
    val_50_data = []
    label2id = {}
    _label = []
    
    for i in label:
        if i not in label2id:
            label2id[i] = len(label2id)
        _label.append(label2id[i])
    
    # 1/3 evaluation set and 2/3training set
    x_train,x_test,train_label,val_label = train_test_split(data,_label,test_size=1/3,random_state=2019)

    # using Tfidfvectorizer to get frequent words as input nodes
    vector =TfidfVectorizer(max_features=50)
    train_50_data = vector.fit_transform(x_train)
    val_50_data = vector.transform(x_test)
    
    # 60
    vector = TfidfVectorizer(max_features=60)
    train_60_data = vector.fit_transform(x_train)
    val_60_data = vector.transform(x_test)
    
    return np.array(train_50_data.todense(), dtype=np.float32), np.array(train_60_data.todense(), dtype=np.float32), train_label, np.array(val_60_data.todense(), dtype=np.float32), np.array(val_50_data.todense(), dtype=np.float32), val_label


# In[ ]:


spam_data, spam_label = spam('../spam.csv')
rt_data, rt_label = rt('../rt-polaritydata/rt-polaritydata/')


# In[264]:


spam_train_50_data, spam_train_60_data, spam_train_label, spam_val_60_data, spam_val_50_data, spam_val_label = handle(spam_data, spam_label)


# In[265]:


rt_train_50_data, rt_train_60_data, rt_train_label, rt_val_60_data, rt_val_50_data, rt_val_label = handle(rt_data, rt_label)


# In[144]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[145]:


class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[244]:


EPOCHS = 10
BATCH_SIZE = 32


# In[258]:


def train(model, train_data, train_label, val_data, val_label):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    max_per = 0
    for epoch in range(EPOCHS):
        for i in range(train_data.shape[0] // BATCH_SIZE + 1):
            data = torch.tensor(train_data[i * BATCH_SIZE: min(len(train_data), (i+1) * BATCH_SIZE)])
            label = torch.tensor(train_label[i * BATCH_SIZE: min(len(train_data), (i+1) * BATCH_SIZE)])
            out = model(data)     
            loss = criterion(out, label)
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()
            count = 0
            for i in range(val_data.shape[0] // BATCH_SIZE + 1):
                data = torch.tensor(val_data[i * BATCH_SIZE: min(len(train_data), (i+1) * BATCH_SIZE)])
                label = val_label[i * BATCH_SIZE: min(len(train_data), (i+1) * BATCH_SIZE)]
                out = model(data)
                prediction = torch.max(out, 1)[1].data.numpy().tolist()
                count += np.sum([i == j for i,j in zip(label, prediction)])
            temp = count / val_data.shape[0] * 100
            max_per = max(temp, max_per)
    return max_per


# In[259]:


model1 = Network(50, 2)
model2 = Network(50, 10)
model3 = Network(60, 2)


# ## spam Dataset

# In[260]:


train(model1, spam_train_50_data, spam_train_label, spam_val_50_data, spam_val_label)


# In[263]:


train(model2, spam_train_50_data, spam_train_label, spam_val_50_data, spam_val_label)


# In[262]:


train(model3, spam_train_60_data, spam_train_label, spam_val_60_data, spam_val_label)


# In[ ]:


model1 = Network(50, 2)
model2 = Network(50, 10)
model3 = Network(60, 2)


# # rt Dataset

# In[270]:


train(model1, rt_train_50_data, rt_train_label, rt_val_50_data, rt_val_label)


# In[277]:


train(model2, rt_train_50_data, rt_train_label, rt_val_50_data, rt_val_label)


# In[282]:


train(model3, rt_train_60_data, rt_train_label, rt_val_60_data, rt_val_label)


# In[ ]:





# In[ ]:




