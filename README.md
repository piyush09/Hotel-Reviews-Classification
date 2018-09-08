This repository is implementation of Naive Bayes Classifier and Perceptron Classifiers (Vanilla and Averaged) to identify hotel reviews as either true or fake, and either positive or negative. In this, the word tokens are used as features for classification.

## **Data Description**

1. "train-labeled.txt" contains labeled training data with a single training instance (hotel review) per line (total 960 lines). First 3 tokens in each line are:
   - unique 7-character alphanumeric identifier
   - label True or Fake
   - label Pos or Neg
The tokens are followed by the text of the review.

2. "dev-text.txt" contains unlabeled development data, with just the unique identifier followed by the text of the review (total 320 lines).

3. "dev-key.txt" contains corresponding labels for the development data, to serve as an answer key.

## **Naive Bayes Classifier Description**

1. nblearn3.py will learn a Naive Bayes Model from the training data, and nbclassify3.py will use the model to classify new data.

2. nblearn3.py - This program learns a Naive Bayes Model, and write the model parameters to a file called "nbmodel.txt".

3. nbclassify3.py - This program will read the parameters of a naive Bayes model from the file "nbmodel.txt", classify each entry in the test data, and write the results to a text file called "nboutput.txt".

## **Perceptron Classifiers (Vanilla and Averaged) Description**

1. Two programs are written - "perceplearn3.py" which learns perceptron models (vanilla and averaged) from the training data, and "percepclassify3.py" which uses the models to classify new data.

2. perceplearn3.py - This program learns perceptron models, and writes the model parameters to two files: "vanillamodel.txt" for the vanilla perceptron, and "averagedmodel.txt" for the averaged perceptron.

3. percepclassify3.py - This program reads the parameters of perceptron model from the model file ("vanillamodel.txt" or "averagedmodel.txt"), classify each entry in the test data, and write the results to a text file called percepoutput.txt
