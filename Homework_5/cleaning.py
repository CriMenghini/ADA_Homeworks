# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:55:49 2016

@author: cristinamenghini
"""

import nltk
from nltk.corpus import stopwords

def clean_entire_corpus(unique_corpus, length, extra):
    """ This function returns the cleaned corpus. 
    It takes as inputs:
    
    @unique_corpus: the entire non preprocessed corpus
    @length: minimum length of the words
    @extra: list of extra word not to include in the corpus"""
    
    # Tokenize
    text = nltk.word_tokenize(unique_corpus)
    
    # For each token: discard it if with length less than @length and get the lowercase
    text_token = [i.lower() for i in text if len(i) >= length]
    
    # Define stopwords list
    stop_words = set(stopwords.words('english') + extra)
    
    # Filter words according to the stopword list 
    filtered_words = [word for word in text_token if word not in stop_words]
    
    # Obtain the new corpus string
    clean_corpus = ' '.join(filtered_words)
    
    return clean_corpus
    