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

