# CNN-yelp-challenge-2016-sentiment-classification
This repository trains a word level Convolutional Neural Network model for sentiment classification task on [Yelp Challenge 2016](https://www.yelp.com/dataset_challenge) using standard deep learning packages.</br>

The task is based on the `yelp_academic_dataset_review.json` file (5 million rows) in the challenge. It has two fields, namely "stars" and "text". The "text" field is customer's raw review sentence, while the "stars" field is the customer's rating for the corresponding review ranging from 1 to 5.</br>

The model architecture is described in `Components` section. In the first layer, I experimented with both `word2vec` and `keras` built-in embedding.</br>

In order to train the model in reasonable time, I first randomly sampled 1 million datapoint, and ended with 399850 samples after removing `null` values. The class distribution of this subset is shown in table 1.</br>

| 1     | 2     | 3     | 4      | 5      |
|-------|-------|-------|--------|--------|
| 46906 | 34283 | 50678 | 106067 | 161916 |
| 11.7% | 8.6%  | 12.7% | 26.5%  | 40.5%  |

I applied the model to both a binary and a multi-lable classification task on this subset.

In the binary setting, reviews with star greater than 2 are regarded as positive samples, otherwise as negative ones. The model achieved 77.91% accuracy on the validation set after 2 epoch training (see `Components` section).</br>

For the multi-label classification task, It achieved ~40% accuracy on test set after 1 epoch training. The result is shown in `train_multi_class.ipynb`.</br>

Feel free to continue my work, and let me know if you achieve better result!

## Requirements
* `Keras`: `pip install keras` (1.0.3)
* `Theano`: `pip install theano` (0.8.0.dev0)

## Components
This repository contains the following components:
* `json-csv.py`</br>This is the data preprocessing script, it converts the `yelp_academic_dataset_review.json` file to a csv file named `review.csv`. The reason is that this `json` file is not valid, but each of its row is. So we have to process it line by line.
* `Word2VecUtility.py`</br>It's borrowed from Kaggle's [word2vec tutorial](https://github.com/wendykan/DeepLearningMovies). It segments a sentence into a word list or a sentence list.
* `word2vec_model.ipynb`</br>It trains a `word2vec` model on the subset reviews. Each word is represented by a 300 dimensional vector. The final model was named as `300features_40minwords_10context`.
* `train_with_word2vec_embedding.ipynb`</br>This file trains a 1D CNN for sentiment classification using `word2vec` embedding. (the embedded data set has a shape of (N, 50, 300), see `Details` section). Unfortunately, my machine was not able to finish the training due to memory issue. So I turn to use Keras' built-in embedding layer instead.
* `train_keras_embedding.ipynb`</br>It trains a model similar to the previous one. The only difference is the embedding layer. The architecture of this model is : Embedding layer - Dropout - Convolution1D - MaxPooling1D - Full Connected layer - Dropout - Relu activation - Sigmoid (with binary cross entropy loss). It was trained on 319880 samples and validated on 79970 samples (train acc: 0.7791 and val_acc: 0.7761 after 2 epoch training).
* `train_multi_class.ipynb`</br>It trains a multi-label classification model with the same architecture on the same subset. It achieved ~40% validation accuracy after 1 epoch training.

## Details
To train CNN models on text data, we need to represent the dataset in 2d matrices (just like traning CNN models on images). There are many ways to achieved this purpose, while in this task, I just tried two apporaches -- one is using `word2vec` embedding, while the other is using keras' built-in embedding layer.</br>

### Word2vec embedding

With a word2vec model, we can transform each review into a fixed length of words with each word represented by its word vector by truncating and padding.

e.g. We can set `max_length` = 50 (max number of words for each review) and the word2vec vocabulary size as 5000.</br> The indices of word in word2vec model are all increased by 3 because 0, 1, 2 are reserved for special purposes. Specifically, reviews with less than 50 word are padded with 0 at the beginning, and longer reviews are truncated to the first 50 words. We let all reviews begin with index 1 and all the words outside of the vocabulary be replaced by 2. Then, for each review, we map its words to the corresponding word vectors. In the end, each review is represented as a (50, 300) matrix.

### Keras embedding layer

Similar to word2vec, just regard Keras embedding layer as an end-to-end trained word embedding.

### Replication

To replication my result, first download the dataset and run `json-csv.py` and `word2vec_model.py` to sample the exact  399850 reviews I used in this task. Run `train_keras_embedding.py` to train a CNN model using keras embedding layer. You can also run `train_with_word2vec_embedding.py` if you want to use word2vec embedding (You need to train a word2vec model beforehand). Make sure you get the sampled data set before you train the model, or you are free to train on all 5 million reviews.</br>

If you would like to predict the 5-category star for each review, refer to `train_multi_class.ipynb`.
