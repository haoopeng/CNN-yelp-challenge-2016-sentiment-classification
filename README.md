# CNN-yelp-challenge-2016-sentiment-classification
This repository illustrates how to train a word level Convolutional Neural Network model for sentiment classification task on [Yelp Challenge 2016](https://www.yelp.com/dataset_challenge).</br>

The model uses the `yelp_academic_dataset_review.json` dataset(5 million rows), which has two fields namely "stars" and "text". The "stars" is the customers' rating ranging from 1 to 5, the "text" field is the customers' raw review sentence.</br>

In the first layer of my CNN architecture, I experimented with both `word2vec` and `keras` built-in embedding as the word embedding layer. In order to train a CNN model on my machine in reasonable time, I sampled 1 million datapoint and ended with 399850 samples after removing `null` values. The class distribution of this subset is in table 1.</br>

| 1     | 2     | 3     | 4      | 5      |
|-------|-------|-------|--------|--------|
| 46906 | 34283 | 50678 | 106067 | 161916 |
|-------|-------|-------|--------|--------|
| 11.7% | 8.6%  | 12.7% | 26.5%  | 40.5%  |

At first, I started with a binary classification task. Review with star greater than 2 is treated as positive sample, otherwise as negative one. The model achieved 77.91% accuracy on the validation set after 2 epoch(see `train_keras_embedding.ipynb`).</br>

I also trained a multi-label classification model using the same architecture on the same data set. I got about 40% validation accuracy after 1 epoch training (it was intended for 2 epoch, but I don't have much time to train it). The result can be checked in `train_multi_class.ipynb`. Feel free to continue my work, please let me know if you get better result. :-)

## Requirements
* `Keras`: `pip install keras`
* `Theano`: `pip install theano`

## Components
This repository contains the following components:
* `json-csv.py` : This is the data preprocessing file, it converts the `yelp_academic_dataset_review.json` file to a `review.csv` in the same directory. Notice that the `json` file is actually not a valid json file that you can load directly, but each its row is. So you have to process it line by line.
* `Word2VecUtility.py` : It is borrowed from Kaggle's [word2vec tutorial](https://github.com/wendykan/DeepLearningMovies). It processes each sentence into word list or sentence list.
* `word2vec_model.ipynb` : It trains a Google's `word2vec` model on all the 1859888 sentences (processed from all the 399850 reviews using `Word2VecUtility.py`. Each word would be represented by a 300 dimensional vector. The final model named `300features_40minwords_10context` is written to disk.
* `train_with_word2vec_embedding.ipynb`: This file trains a 1D CNN for sentiment classification using `word2vec` embedding. (the embedded data set has a shape of (399850, 50, 300), see `Details` sections). Unfortunately, my machine was not able to train it due to the memory size. So I turn to use Keras' built-in embedding layer instead.
* `train_keras_embedding.ipynb`: This trains a model similar to the previous one. The only difference is on the embedding layer. The architecture of this model is : Embedding layer - Dropout - Convolution1D - MaxPooling1D - Full Connected layer - Dropout - Relu activation - Sigmoid (with binary cross entropy loss). Other model parameters are easy to be seen in this file. It is trained on 319880 samples and validated on 79970 samples. After 16 hours' training (2 epoch), I got train acc: 0.7791 and val_acc: 0.7761.
* `train_multi_class.ipynb`: It trains a multi-label classification model(the labels are namely: 1, 2, 3, 4, 5 stars) with the same architecture on the same subset. I got about 40% accuracy after 1 epoch training.

## Details
In word level CNN models, one main task is to get the embeded data set (thus you can perform convolution on it!). In this project, I tried two different apporaches for this. One is using `word2vec`, another is using keras' embedding layer.</br>

### Word2vec embedding

For this method, I need to train 2 `word2vec` model first. After that, we can transform each review into a fixed length of words with each word represented by its word2vec vector. In order to train the model on my machine, I set `max_length` = 50(max number of words for each review, those with less than 50 word are padded with 0), `max_word` = 5000 (max number of word features in the word2vec model).</br>

To do this, I first built a word index dictionary to represent each review as a list of word indexes. The indexes of word in word2vec model are all increased by 3. I let all reviews begin with the index 1 and all the words whose indexes are outside of [3 and 5003] be replaced by 2. Then, for each review, I map its words to the corresponding word2vec vectors, which mean each review is now a (50, 300) matrix.

### Keras embedding layer

To use keras embedding layer, I feed in the review data as its word indexes representation(list of list of word indexed) as we did in `train_with_word2vec_embedding.ipynb`. I set the `embedding_dims` = 100. To be more specific, the shape of input data is (399850, 50), the output of embedding lay has a shape of (399850, 50, 100).

### Replication

To replication my result, you should first download the data set and run `json-csv.py`. Then run `word2vec_model.py` to sample the 399850 reviews. You can continue to train a word2vec model on this sampled data set if you want to use word2vec embedding (It took my mac 4 mins to train).</br>

Run `train_keras_embedding.py` to train a CNN using keras embedding layer. You can also run `train_with_word2vec_embedding.py` if you want (this requires a lot of computation resource). Make sure you get the sampled data set before you train the model (otherwise you can modify my code to train on 5 million reviews!).</br>

If you would like to predict the exact star(out of 1, 2, 3, 4, 5) for each review, please try `train_multi_class.ipynb`.
