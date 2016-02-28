
'''
train cnn mode for sentiment classification on yelp data set
author: hao peng
'''
import pickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from Word2VecUtility import Word2VecUtility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def get_volcabulary_and_list_words(data):
    reviews_words = []
    volcabulary = []
    for review in data["text"]:
        review_words = Word2VecUtility.review_to_wordlist(
            review, remove_stopwords=True)
        reviews_words.append(review_words)
        for word in review_words:
            volcabulary.append(word)
    volcabulary = set(volcabulary)
    return volcabulary, reviews_words

def get_reviews_word_index(reviews_words, volcabulary, max_words, max_length):
    word2index = {word: i for i, word in enumerate(volcabulary)}
    # use w in volcabulary to limit index within max_words
    reviews_words_index = [[start] + [(word2index[w] + index_from) for w in review] for review in reviews_words]
    # in word2vec embedding, use (i < max_words + index_from) because we need the exact index for each word, in order to map it to its vector. And then its max_words is 5003 instead of 5000.
    reviews_words_index = [[i if (i < max_words) else oov for i in index] for index in reviews_words_index]
    # padding with 0, each review has max_length now.
    reviews_words_index = sequence.pad_sequences(reviews_words_index, maxlen=max_length, padding='post', truncating='post')
    return reviews_words_index


# data processing para
max_words = 5000
max_length = 50

# model training parameters
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

# index trick parameters
index_from = 3
start = 1
# padding = 0
oov = 2
'''
data = pd.read_csv(
    'review_sub_399850.tsv', header=0, delimiter="\t", quoting=3, encoding='utf-8')
print('get volcabulary...')
volcabulary, reviews_words = get_volcabulary_and_list_words(data)
print('get reviews_words_index...')
reviews_words_index = get_reviews_word_index(reviews_words, volcabulary, max_words, max_length)

print reviews_words_index[:20, :12]
print reviews_words_index.shape

labels = data["stars"]
labels[labels <= 3] = 0
labels[labels > 3] = 1

pickle.dump((reviews_words_index, labels), open("399850by50reviews_words_index.pkl", 'wb'))
'''
(reviews_words_index, labels) = pickle.load(open("399850by50reviews_word2vec_words_index.pkl", 'rb'))

index = np.arange(reviews_words_index.shape[0])
train_index, valid_index = train_test_split(
    index, train_size=0.8, random_state=520)

train_data = reviews_words_index[train_index]
valid_data = reviews_words_index[valid_index]
train_labels = labels[train_index]
valid_labels = labels[valid_index]
print train_data.shape
print valid_data.shape

del(labels, train_index, valid_index)

print "start training model..."

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_words + index_from, embedding_dims, \
                    input_length=max_length))
model.add(Dropout(0.25))

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
