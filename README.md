This repository is a Naive Bayes Classifier to identify hotel reviews as either true or fake, and either positive or negative. In this, the word tokens are used as features for classification.

## **Data Description**

1. "train-labeled.txt" contains labeled training data with a single training instance (hotel review) per line (total 960 lines). First 3 tokens in each line are:
   - unique 7-character alphanumeric identifier
   - label True or Fake
   - label Pos or Neg
The tokens are followed by the text of the review.

2. "dev-text.txt" contains unlabeled development data, with just the unique identifier followed by the text of the review (total 320 lines).

3. "dev-key.txt" contains corresponding labels for the development data, to serve as an answer key.

## **Code Description**

1. nblearn3.py will learn a Naive Bayes Model from the training data, and nbclassify3.py will use the model to classify new data.

2. nblearn3.py - This program learns a Naive Bayes Model, and write the model parameters to a file called "nbmodel.txt".

3. nbclassify3.py - This program will read the parameters of a naive Bayes model from the file "nbmodel.txt", classify each entry in the test data, and write the results to a text file called "nboutput.txt".
