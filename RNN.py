#!/usr/bin/env python
# coding: utf-8

# In[13]:


from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential

from sklearn.metrics import precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("../spam.csv")


# In[3]:


texts = []
labels = []
for i, label in enumerate(data['Category']):
    texts.append(data['Message'][i])
    if label == 'ham':
        labels.append(0)
    else:
        labels.append(1)

texts = np.asarray(texts)
labels = np.asarray(labels)


print("number of texts :" , len(texts))
print("number of labels: ", len(labels))


# In[4]:


from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# number of words used as features
max_features = 10000
# cut off the words after seeing 500 words in each document(email)
maxlen = 500


# we will use 80% of data as training, 20% as validation data
training_samples = int(5572 * .8)
validation_samples = int(5572 - training_samples)
# sanity check
print(len(texts) == (training_samples + validation_samples))
print("The number of training {0}, validation {1} ".format(training_samples, validation_samples))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)

print("data shape: ", data.shape)

np.random.seed(42)
# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


texts_train = data[:training_samples]
y_train = labels[:training_samples]
texts_test = data[training_samples:]
y_test = labels[training_samples:]


# In[14]:


model = Sequential()
model.add(Embedding(max_features, 32))

# use simple RNN
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# use adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)


# In[15]:


acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[16]:


# predict test dataset
pred = model.predict_classes(texts_test)
acc = model.evaluate(texts_test, y_test)


# In[17]:


# get the precision, recall and F1 score
# precision
precision = precision_score(pred, y_test, average='binary')
# recall
recall = recall_score(pred, y_test, average='binary')
# F1 score
F1_score = f1_score(pred, y_test, average='binary')


# In[18]:


# print
print("Test presicion is {:.4f}  ".format(precision))
print("Test recall is {:.4f}  ".format(recall))
print("Test F1 score is {:.4f}  ".format(F1_score))


# In[ ]:




