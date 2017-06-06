# Sentiment_Analysis_Doc2Vec
Sentiment Analysis of Tweets and IMDb dataset review using Traditional NLP and Doc2Vec Technique

#### Project Summary
In this project, we performed sentiment analysis over IMDB movie reviews and Twitter data. Our goal was to classify tweets or movie reviews as either positive or negative. Towards the end, we were able to be givenlabeled training to build the model and labeled testing data to evaluate the model. For classification, we experimented with [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) as well as a [Naive Bayes classifier](http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes) from python’s wellregarded machine learning package [scikit-learn](http://scikit-learn.org/stable/index.html). As a point of reference, [Stanfords Recursive Neural Network](https://nlp.stanford.edu/sentiment/code.html) code produced an accuracy of 51.1% on the IMDB dataset and 59.4% on the Twitter data.

A major part of this project was the task of generating feature vectors for use in these classifiers. 
We explored two methods:
 - A more traditional NLP technique where the features are simply “important” words and the feature vectors are simple binary vectors and 
 - The Doc2Vec technique where document vectors are learned via artificial neural networks (a summary can be found [here](https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis)).

#### Project Setup
The python packages that you will need for this project are scikitlearn, nltk, and gensim. To install these, simply use the pip installer sudo pip install X or, if you are using Anaconda, conda install X , where X is the package name.

#### Datasets
The IMDB reviews and tweets can be found in the data folder. These have already been divided into train and test sets.
- The IMDB dataset, originally found [here](http://ai.stanford.edu/~amaas/data/sentiment/), that contains 50,000 reviews split evenly into 25k train and 25k test sets. Overall, there are 25k pos and 25k neg reviews. In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets.
- The Twitter Dataset, taken from [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/), contains 900,000 classified tweets split into 750k train and 150k test sets. The overall distribution of labels is balanced (450k pos and 450k neg).

#### Project Requirements
- feature_vecs_NLP: The comments in the code should provide enough instruction. Just keep in mind that a word should be counted at most once per tweet/review even if the word has occurred multiple times in that tweet/review.
- build_models_NLP: Refer to the documentation linked above for details on how to call the functions.
- feature_vecs_DOC: Some documentation for the doc2vec package can be found [here](https://radimrehurek.com/gensim/models/doc2vec.html). The first thing you will want to do is make a list of LabeledSentence objects from the word lists. These objects consist of a list of words and a list containing a single string label. You will want to use a different label for the train/test and pos/neg sets. For example, we used TRAIN_POS_i, TRAIN_NEG_i, TEST_POS_i, and TEST_NEG_i, where i is the line number. [This blog](https://rare-technologies.com/doc2vec-tutorial/) may be a helpful reference.
- build_models_DOC: Similar to the other function.
- evaluate_model: Here you will have to calculate the true positives, false positives, true negatives, false negatives, and accuracy.

#### Output
[!Alt text](imgs/imdb0.png "NLP Output on IMDB Dataset")




