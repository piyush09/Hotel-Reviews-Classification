import sys
import os
import math
import time
import json
from pathlib import Path

fake_dict = dict()
true_dict = dict()
pos_dict = dict()
neg_dict = dict()

fake_class_count = 0
true_class_count = 0
pos_class_count = 0
neg_class_count = 0

fake_word_count = 0
true_word_count = 0
pos_word_count = 0
neg_word_count = 0

vocabulary = set()
# Removing stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
punctuation = ['<', '>', '?', '.', '"', ')', '(', '|', '-', '#', '*', '+', ';', '!', '/', '\\', '=', ',', ':', '$','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '@', '&', '%', '{', '}', '^', '~']

data = open("F:/USC Courses/Spring 2018/CSCI 544/Programming Exercises/Programming Exercise 2/coding-2-data-corpus/train-labeled.txt",encoding="UTF-8").read().strip().splitlines()


for line in data:  # Reading each line in the file


    for i in range(0,len(punctuation)):
        line.replace(punctuation[i], ' ')

    token_list = line.split()  # Token list containing tokens in each line
    token_list = [item.lower() for item in token_list] # lowering the words in the token list

    for word in list(token_list):

        #if word in stopwords:  # removing stop words from the token list
        #    token_list.remove(word)
        for word in token_list[3:]:
            vocabulary.add(word)

    if token_list[1] == "fake":
        fake_class_count += 1  # Calculating class count (Number of lines in input with fake class)
        for token in token_list[3:]:  # Accessing tokens from 3rd element till last element in line
            fake_word_count += 1    # Calculating total number of words in fake class

            if token in fake_dict:
                fake_dict[token] += 1
            else:
                fake_dict[token] = 1

    if token_list[1] == "true":
        true_class_count += 1
        for token in token_list[3:]:  # Accessing tokens from 3rd element till last element in line
            true_word_count += 1     # Calculating total number of words in true class
            if token in true_dict:
                true_dict[token] += 1
            else:
                true_dict[token] = 1

    if token_list[2] == "neg":
        neg_class_count += 1
        for token in token_list[3:]:  # Accessing tokens from 3rd element till last element in line
            neg_word_count += 1  # Calculating total number of words in negative class
            if token in neg_dict:
                neg_dict[token] += 1
            else:
                neg_dict[token] = 1

    if token_list[2] == "pos":
        pos_class_count += 1
        for token in token_list[3:]:  # Accessing tokens from 3rd element till last element in line
            pos_word_count += 1  # Calculating total number of words in positive class
            if token in pos_dict:
                pos_dict[token] += 1
            else:
                pos_dict[token] = 1

prior_class_probability_dict = {}
prior_class_probability_dict['fake'] = math.log(1.0 *(fake_class_count / (fake_class_count + true_class_count)))
prior_class_probability_dict['true'] = math.log(1.0 *(true_class_count / (fake_class_count + true_class_count)))
prior_class_probability_dict['positive'] = math.log(1.0 *(pos_class_count / (pos_class_count + neg_class_count)))
prior_class_probability_dict ['negative'] = math.log(1.0 *(neg_class_count / (pos_class_count + neg_class_count)))

vocabularycount = len(vocabulary)

for token in vocabulary:

    # division(log subtraction) of probabilities of word belonging to fake class/total no of words in fake class
    fake_dict[token] = math.log(float(fake_dict.get(token,0) + 1) / (fake_word_count + vocabularycount))

    true_dict[token] = math.log(float(true_dict.get(token,0) + 1) / (true_word_count + vocabularycount))

    pos_dict[token] = math.log(float(pos_dict.get(token,0) + 1) / (pos_word_count + vocabularycount))

    neg_dict[token] = math.log(float(neg_dict.get(token,0) + 1) / (neg_word_count + vocabularycount))

output = open('nbmodel.txt', 'w')   # Writing all contents to file
result = dict()
result['fake_Dictionary'] = fake_dict
result['true_Dictionary'] = true_dict
result['positive_Dictionary'] = pos_dict
result['negative_Dictionary'] = neg_dict
result['prior_class_probability'] = prior_class_probability_dict

json.dump(result,output)
output.close()

