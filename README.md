# yelp-challenge-2016-DeepLearning
This repository is the code for training a word level CNN for sentiment classification on [Yelp Challenge 2016 dataset](https://www.yelp.com/dataset_challenge). The code deals with the `yelp_academic_dataset_review.json` sub dataset, which has two columns namely "stars" and "text", and it contains about 5 million rows. The "stars" is the customers' rating from 1 to 5, the "text" field is the customers' review sentence. 

In order to train the model on my local machine in reasonal time, I subsampled 1 million from it and get 399850 rows after removing rows with `null` values. The model achieved 77.91% accuracy on the validation set after 2 epoch, which you can chech from the `train_keras_embedding.ipynb` file.

## Components
This repository contains the following components:
* `json-csv.py` : This is the data preprocessing file, it converts the `yelp_academic_dataset_review.json` file to a `review.csv` in the same directory. Notice that the `json` file is actually not a valid json file that you can load directly, but each its row is. So you have to process it line by line.
* `Word2VecUtility.py` : This file is borrowed from Kaggle's [word2vec tutorial](https://github.com/wendykan/DeepLearningMovies). It processes each sentence into word list or sentence list.
* `word2vec_model.ipynb` : This is used to train a Google's `word2vec` model on all the 1859888 sentences(processed from all the 399850 reviews using `Word2VecUtility.py`. The final model named `300features_40minwords_10context` is written to disk.
