import sys
import os
import math
import time
import json

fake_prob = 0
true_prob = 0
pos_prob = 0
neg_prob = 0

model_file = open('nbmodel.txt', 'r')
modelResult = json.load(model_file)

target = open('nboutput.txt', 'w',encoding="UTF-8")

fake_dict = modelResult['fake_Dictionary']
true_dict = modelResult['true_Dictionary']
pos_dict = modelResult['positive_Dictionary']
neg_dict = modelResult['negative_Dictionary']
prior_class_probability_dict = modelResult['prior_class_probability']


# Removing stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
punctuation = ['<', '>', '?', '.', '"', ')', '(', '|', '-', '#', '*', '+', ';', '!', '/', '\\', '=', ',', ':', '$','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '@', '&', '%', '{', '}', '^', '~']

developmentlines = open("F:/USC Courses/Spring 2018/CSCI 544/Programming Exercises/Programming Exercise 2/coding-2-data-corpus/dev-text.txt",encoding="UTF-8").readlines()

for line in developmentlines:
    for i in range(0,len(punctuation)):
        line.replace(punctuation[i], ' ')
    token_list = line.split()
    key = token_list[0]
      # token_list is the list of tokens in each line in file
    token_list = [item.lower() for item in token_list]  # lowering the words in the token list


    for word in token_list:  # removing stop words from the token list
        if word in stopwords:
            token_list.remove(word)

    fake_prob = 0
    true_prob = 0
    pos_prob = 0
    neg_prob = 0

    for token in token_list[1:]: # iterating over every token in token_list

        fake_prob += fake_dict.get(token,0)

        true_prob += true_dict.get(token,0)

        pos_prob += pos_dict.get(token,0)

        neg_prob += neg_dict.get(token,0)

    # Multiplication(log addition) of prior probability of fake class
    fake_prob += prior_class_probability_dict['fake']
    true_prob += prior_class_probability_dict['true']
    pos_prob += prior_class_probability_dict['positive']
    neg_prob += prior_class_probability_dict['negative']

    if true_prob >= fake_prob:
        class1 = "True"
    else:
        class1 = "Fake"

    if pos_prob >= neg_prob:
        class2 = "Pos"
    else:
        class2 = "Neg"

    target.write(key.strip() + " " + class1.strip() + " " + class2.strip())
    target.write('\n')
