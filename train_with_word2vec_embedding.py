
# coding: utf-8

# In[26]:

'''
train cnn mode for sentiment classification on yelp data set
author: hao peng
'''
import pickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from Word2VecUtility import Word2VecUtility
from gensim.models import word2vec
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D

model = word2vec.Word2Vec.load("300features_40minwords_10context")
print model.syn0.shape
print model["chinese"]
print model.doesnt_match("man woman child kitchen".split())
print model.doesnt_match("coffee tea juice restaurant".split())
print model.most_similar("delicious")
print model.most_similar("chinese")

# data embedding parameters
max_length = 50
max_words = 5000
# max_words = model.syn0.shape[0]
num_features = 300

# model training parameters
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

# index trick parameters
index_from = 3
# padding = 0
start = 1
oov = 2

words_set = set(model.index2word)
word2index = { word : (i + index_from) for i,word in enumerate(words_set) }
index2word = { i : word for word, i in word2index.items() }
index2word[0] = '0'
index2word[1] = '1'
index2word[2] = '2'
# 'Word2Vec' object does not support item assignment
padding_model = {}
padding_model['0'] = np.random.standard_normal(num_features)
padding_model['1'] = np.random.standard_normal(num_features)
padding_model['2'] = np.random.standard_normal(num_features)

data = pd.read_csv('review_sub_399850.tsv', header=0, delimiter="\t", quoting=3, encoding='utf-8')

reviews_words = []
for review in data["text"]:
    review_words = Word2VecUtility.review_to_wordlist(review, remove_stopwords = True)
    # each word index has already been increased by 3.
    review_words = [start] + [word2index[w] if (w in model) else oov for w in review_words]
    # index from 0,1,... to 5002
    review_words = [oov if (ix >= (max_words + index_from)) else ix for ix in review_words]
    reviews_words.append(review_words)

# padding with 0, each review has max_length now.
reviews_words = sequence.pad_sequences(reviews_words, maxlen = max_length, padding='post', truncating='post')

print reviews_words[:20, :12]
print reviews_words.shape

labels = data["stars"]
# print labels[:10], labels.shape
labels[labels <= 3] = 0
labels[labels > 3] = 1
# print labels[:10]
# print (labels == 0).sum()

pickle.dump((reviews_words, labels), open("399850by50reviews_word2vec_words_index.pkl", 'wb'))

# (reviews_words, labels) = pickle.load(open("399850by50reviews_word2vec_words_index.pkl", 'rb'))

index = np.arange(reviews_words.shape[0])
train_index, valid_index = train_test_split(index, train_size = 0.8, random_state = 520)



# In[43]:

data_matrix = np.empty((reviews_words.shape[0], max_length, num_features))
for i in xrange(0, reviews_words.shape[0]):
    data_matrix[i,:,:] = np.array([model[index2word[ix]] if (index2word[ix] in model) else padding_model[index2word[ix]] for ix in reviews_words[i]])

del(reviews_words, model, padding_model, word2index, index2word)

# pickle.dump((data_matrix, labels), open("399850by50by300.pkl", 'wb'))
# (data_matrix, labels) = pickle.load(open("399850by50by300.pkl", 'rb'))

train_data = data_matrix[train_index]
valid_data = data_matrix[valid_index]
train_labels = labels[train_index]
valid_labels = labels[valid_index]
print train_data.shape
print valid_data.shape

del(data_matrix)
del(labels, train_index, valid_index)

print "start training model..."

model = Sequential()
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:

# filter_length is like filter size, subsample_length is like step in 2D CNN.
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')
model.fit(train_data, train_labels, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(valid_data, valid_labels))

