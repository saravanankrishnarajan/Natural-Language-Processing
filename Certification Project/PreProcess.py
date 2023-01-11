# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:14:09 2023

@author: Saravanan
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def Tokenize(txt):
    word_txt = word_tokenize(txt)
    return word_txt

def RemoveStopWords(tokenized_words):
    StopWords = list(stopwords.words('english'))
    post_stopwords = []
    [post_stopwords.append(i) for i in tokenized_words if i not in StopWords]

    return post_stopwords

def Lemmatize(tokenized_words):
    Lemm = WordNetLemmatizer()
    for i in range(0, len(tokenized_words)):
        tokenized_words[i] = Lemm.lemmatize(tokenized_words[i])

    return tokenized_words

def refine(txt):
    tokenized_word = Tokenize(txt)
    post_stopwords = RemoveStopWords(tokenized_word)
    post_lemmatize = Lemmatize(post_stopwords)
    refined = post_lemmatize

    return refined

#txt = """Gold is a chemical element with symbol Au (from Latin: aurum) and atomic number 79, making it one of the higher atomic number elements that occur naturally. In its purest form, it is a bright, slightly reddish yellow, dense, soft, malleable, and ductile metal. Chemically, gold is a transition metal and a group 11 element. It is one of the least reactive chemical elements and is solid under standard conditions. Gold often occurs in free elemental (native) form, as nuggets or grains, in rocks, in veins, and in alluvial deposits. It occurs in a solid solution series with the native element silver (as electrum) and also naturally alloyed with copper and palladium. Less commonly, it occurs in minerals as gold compounds, often with tellurium (gold tellurides)."""

#user_input = input("Enter string as input")

#print(refine(user_input))
