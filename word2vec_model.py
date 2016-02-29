
# coding: utf-8

# In[1]:

from __future__ import division
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Word2VecUtility import Word2VecUtility
import pickle
import pandas as pd
import numpy as np
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)


# In[2]:

data = pd.read_csv('review.tsv', header=0, delimiter="\t", quoting=3)
print '\nThe first review is:\n'
print data["text"][0], '\n'
print data.shape
print data.columns


# In[30]:

print data['stars'][:3]
print
print data.ix[:2]['text']


# In[3]:


size = 1000000 #80000
subdata = data.sample(n = size, random_state=520)
# some review's text field is null.
subdata = subdata[pd.notnull(subdata['text'])]
print subdata.index
subdata.to_csv('review_sub_399850.tsv', index=False, quoting=3, sep='\t', encoding='utf-8')


# In[4]:

del(data)
data = subdata
del(subdata)


In[20]:



data = pd.read_csv('review_sub_399850.tsv', header=0, delimiter="\t", quoting=3, encoding='utf-8')


# In[5]:

print data.shape
print data.columns
print data.index
# data.ix only applies to the newly readin pd data frame.
# iloc applies to both cases(even when you sampled the data)
print data.iloc[:5]


# In[6]:

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[7]:

# print data.ix[0:10]
print data.iloc[:10]['text']
# print data['text'][2]


# In[8]:

review_sents = []
print "Cleaning and parsing the reviews...\n"
for i in xrange( 0, len(data["text"])):
    # sent_reviews += Word2VecUtility.review_to_sentences(data["text"][i], tokenizer)
    review_sents += Word2VecUtility.review_to_sentences(data.iloc[i]["text"], tokenizer)


# # In[53]:

out = open('review_sents_1859888.pkl', 'wb')
pickle.dump(review_sents, out)
out.close()



# # In[11]:

review_sents = pickle.load(open('review_sents_1859888.pkl', 'rb'))
print len(review_sents)
print review_sents[:5]


# In[57]:

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training model..."
model = word2vec.Word2Vec(review_sents, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)


# In[58]:

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)
