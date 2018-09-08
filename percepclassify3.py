import sys
import os
import math
import time
import json
from collections import defaultdict

# model_file = open('vanillamodel.txt', 'r')
model_file = open(sys.argv[1], 'r')

# modelResult = json.load(model_file)

target = open('percepoutput.txt', 'w',encoding="UTF-8")
tf_wt_dict = defaultdict(int)
pn_wt_dict = defaultdict(int)

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
# stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
punctuation = ['<', '>', '?', '.', '"', ')', '(', '|', '-', '#', '*', '+', ';', '!', '/', '\\', '=', ',', ':', '$','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '@', '&', '%', '{', '}', '^', '~']


# tf_wt_dict = modelResult['tf_weight_Dictionary']
# pn_wt_dict = modelResult['pn_weight_Dictionary']
# tf_bias = modelResult['true_fake_bias']
# pn_bias = modelResult['positive_negative_bias']
tfFlag=  False
pnFlag = False
pn = 0
tf = 0

for line in model_file:

    # for i in range(0, len(punctuation)):
    #     line.replace(punctuation[i], ' ')  # Replacing punctuation

    line = line.strip()

    if line == "FIRST_CLASSIFY":
        tfFlag =True
        pnFlag = False
        tf = 1
        continue

    if line == "SECOND_CLASSIFY":
        tfFlag = False
        pnFlag = True
        pn = 1
        continue

    if tfFlag and tf == 1:

        tf+=1
        tf_bias  =  float(line)
        continue

    if tfFlag:
        tokens = line.split()
        tfword,tfval = tokens[0],tokens[1]
        tf_wt_dict[tfword] = float(tfval)
        continue


    if pnFlag and pn == 1:
        pn+=1
        pn_bias  =  float(line)
        continue

    if pnFlag :
        tokens = line.split()
        pnword,pnval = tokens[0],tokens[1]
        pn_wt_dict[pnword] = float(pnval)
        continue


# developmentlines = open("/home/piyush/Desktop/coding-2-data-corpus/dev-text.txt",encoding="UTF-8").readlines()
developmentlines = open(sys.argv[2],'r',encoding="UTF-8").readlines()

res=[]

for line in developmentlines: # Reading each line in the file


    tokens_list = [item.lower() for item in line.split()]

    # for word in tokens_list[3:]:  # Access starting from 3rd word in each line in corpus
    #     if word in stopwords:  # removing stop words from the token list
    #         tokens_list.remove(word)


    temp_word_count= defaultdict(int)

    key=line.split()[0]

    tf_activation = tf_bias
    for token in tokens_list[1:]:

        if token in tf_wt_dict.keys():
            tf_activation += tf_wt_dict[token]

    # print(tf_activation)

    pn_activation = pn_bias
    for token in tokens_list[1:]:
        if token in pn_wt_dict.keys():
            pn_activation += pn_wt_dict[token]

    if tf_activation <= 0:
        tf="Fake"
    else:
        tf="True"

    if pn_activation <= 0:
        pn="Neg"
    else:
        pn="Pos"
    res.append([key,tf,pn])

for item in res:

    target.write(item[0] +" " + item[1] +" " +item[2]+"\n")

target.close()
