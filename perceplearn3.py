import sys
import os
import math
import time
import json
import string
from pathlib import Path
from collections import defaultdict


tf_wt_dict = defaultdict(int)  #True-Fake weight default dictionary
pn_wt_dict = defaultdict(int)  #Positive-Negative weight default dictionary

counter_tf = 1
counter_pn = 1

total_tf_wt_dict = defaultdict(int)
total_pn_wt_dict = defaultdict(int)

total_tf_bias = 0
total_pn_bias = 0


tf_activation = 0 #Iniializing true fake biclassifier activation as 0
pn_activation = 0 #Initializing positive negative biclassifier activation as 0

tf_bias = 0 #Initializing true fake bias as 0
pn_bias = 0 #Initializing positive negative bias as 0

# stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as","at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
#               "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
#               "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
#               "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
#               "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor",
#               "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
#               "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that",
#               "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
#               "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
#               "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
#               "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
#               "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
#               "yourself", "yourselves"]
stopwords = ['a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','the','to','was','were','with','will'];
#stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
punctuation = ['<', '>', '?', '.', '"', ')', '(', '|', '-', '#', '*', '+', ';', '!', '/', '\\', '=', ',', ':', '$','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '@', '&', '%', '{', '}', '^', '~']

# data = open("/home/piyush/Desktop/coding-2-data-corpus/train-labeled.txt",encoding="UTF-8").read().strip().splitlines()
data = open(sys.argv[1],encoding="UTF-8").read().strip().splitlines()

for _ in range(50):

    for line in data: # Reading each line in the file

        lc = 0
        temp_word_count_dict = defaultdict(int)  # Temporary word count default dictionary for each sentence

        tf_activation = 0
        pn_activation = 0


        # for i in range(0,len(punctuation)):
        #     line.replace(punctuation[i], ' ')  # Replacing punctuation


        token_list = [item.lower() for item in line.split()] # lowering and splitting to get the words in token list
        tf_count = 0   # Initializing true-fake count as 0
        pn_count = 0   # Initializing positive-negative count as 0

        for word in token_list[3:]: # Access starting from 3rd word in each line in corpus

            if word in stopwords: # removing stop words from the token list
                token_list.remove(word)

            if token_list[1] == "fake" and tf_count == 0:  # Accessing Fake sentence
                tf_count += 1
                for token in token_list[3:]:  # Accessing tokens from 3rd element till last element in line

                    tf_activation += tf_wt_dict[token] # Adding weight of each token encountered in true-fake activation variable
                    temp_word_count_dict[token]  += 1 # Adding 1 in count of temporary word count dictionary of each token

                tf_activation += tf_bias # Adding true-fake bias value in true-fake activation variable at end of sentence

                # Updating Weights and Bias values
                if ((-1)*tf_activation) <=0: # Case where (y*activation) <= 0

                    for token in temp_word_count_dict:
                        tf_wt_dict[token] += ((-1)* temp_word_count_dict[token])  #Updating weight
                        total_tf_wt_dict[token] += ((-1) * counter_tf * temp_word_count_dict[token])
                    tf_bias += (-1)
                    total_tf_bias += (-1) * counter_tf

            if token_list[1] == "true" and tf_count == 0:  # Accessing True sentence
                tf_count += 1
                for token in token_list[3:]:  # Accessing tokens from 3rd element till last element in line

                    tf_activation += tf_wt_dict[token]
                    temp_word_count_dict[token] += 1  # Adding 1 in count of temporary word count dictionary of each token

                tf_activation += tf_bias  # Adding true-fake bias value in true-fake activation variable at end of sentence

                # Updating Weights and Bias values
                if ((+1) * tf_activation) <= 0:  # Case where (y*activation) <= 0
                    # print("No")
                    for token in temp_word_count_dict:
                        tf_wt_dict[token] += ((+1) * temp_word_count_dict[token])  # Updating weight
                        total_tf_wt_dict[token] += ((+1) * counter_tf * temp_word_count_dict[token])
                    tf_bias += (+1)
                    total_tf_bias += (+1) * counter_tf

            if token_list[2] == "neg" and pn_count == 0:
                pn_count += 1
                for token in token_list[3:]:

                    pn_activation += pn_wt_dict[token]

                pn_activation += pn_bias

                # Updating Weights and Bias values
                if ((-1) * pn_activation) <= 0:  # Case where (y*activation) <= 0
                    for token in temp_word_count_dict:
                        pn_wt_dict[token] += ((-1) * temp_word_count_dict[token])  # Updating weight
                        total_pn_wt_dict[token] += ((-1) * counter_pn * temp_word_count_dict[token])
                    pn_bias += (-1)
                    total_pn_bias += (-1) * counter_pn

            if token_list[2] == "pos" and pn_count == 0:
                pn_count += 1
                for token in token_list[3:]:

                    pn_activation += pn_wt_dict[token]

                pn_activation += pn_bias

                # Updating Weights and Bias values
                if ((+1) * pn_activation) <= 0:  # Case where (y*activation) <= 0
                    for token in temp_word_count_dict:
                        pn_wt_dict[token] += ((+1) * temp_word_count_dict[token])  # Updating weight
                        total_pn_wt_dict[token] += ((+1) * counter_pn * temp_word_count_dict[token])
                    pn_bias += (+1)
                    total_pn_bias += (+1) * counter_pn

            counter_tf += 1
            counter_pn += 1


# output = open('vanillamodel.txt', 'w')   # Writing all contents to file
# result = dict()
# result['tf_weight_Dictionary'] = tf_wt_dict
# result['pn_weight_Dictionary'] = pn_wt_dict
# result['true_fake_bias'] = tf_bias
# result['positive_negative_bias'] = pn_bias
#
# json.dump(result,output)

vModel = open("vanillamodel.txt","w+",encoding="utf8")
vModel.write("FIRST_CLASSIFY\n")

vModel.write(str(tf_bias)+"\n")
for keys in tf_wt_dict.keys():
    vModel.write(keys + " " +str(tf_wt_dict[keys])+"\n")

vModel.write("SECOND_CLASSIFY\n")
vModel.write(str(pn_bias)+"\n")
for keys in pn_wt_dict.keys():
    vModel.write(keys + " " +str(pn_wt_dict[keys])+"\n")
vModel.close()

#print(tf_bias)
#print(pn_bias)

for key in total_tf_wt_dict.keys():
    tf_wt_dict[key] = tf_wt_dict[key] - total_tf_wt_dict[key]/counter_tf
tf_bias = tf_bias - total_tf_bias/counter_tf

for key in total_pn_wt_dict.keys():
    pn_wt_dict[key] = pn_wt_dict[key] - total_pn_wt_dict[key]/counter_pn
pn_bias = pn_bias - total_pn_bias/counter_pn

aModel = open("averagedmodel.txt","w+",encoding="utf8")
aModel.write("FIRST_CLASSIFY\n")
aModel.write(str(tf_bias)+"\n")
for keys in tf_wt_dict.keys():
    aModel.write(keys + " " +str(tf_wt_dict[keys])+"\n")

aModel.write("SECOND_CLASSIFY\n")
aModel.write(str(pn_bias)+"\n")
for keys in pn_wt_dict.keys():
    aModel.write(keys + " " +str(pn_wt_dict[keys])+"\n")

aModel.close()
