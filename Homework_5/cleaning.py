# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:55:49 2016

@author: cristinamenghini
"""

import nltk
import string
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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
    
    
def clean_body_country(x, stop_words, extension, length):
    """ This function returns the list of cleaned list of words in an email.
    It takes as inputs:
    @x: string
    @stopwords: list of stopwords
    @extension: list to exted the stopwords
    @length: minimum length of a  word"""
    
    # Tokenize
    text = nltk.word_tokenize(x)
    
    # Replace the comma adding space and go to lowercase
    replace_comma = [(i.replace(',', ', ')).lower() for i in text]
    
    # Get back to the string
    join_list = " ".join(replace_comma)
    
    # Remove all the punctuation
    for c in string.punctuation:
        join_list = join_list.replace(c,"")
    
    text_new = nltk.word_tokenize(join_list)
    get_rid = set(list(stop_words) + extension)
    list_words = [i for i in text_new if i not in get_rid and len(i) >= length]
    
    return list_words
    

def country_names(list_country):
    """ This function returns a dictionary (key,value):(name_country:[listi other codes]) and a set of all the possible
    ways to express each country.
    It takes as input:
    
    @list_country: a list of objects that stores the country codes."""
    
    # Initialize the two ouputs
    country_dictionary = {}
    country_set = []
    
    # For each object in the country list
    for country in list_country:
        
        # We add an instance to the dictionary with key: name of country - lowercase and value the list
        # of other codes used to identify the country.
        country_dictionary[(country.name).lower()] = [(country.alpha_2).lower(), (country.alpha_3).lower()]

        # For some state we add manually other way to express them
        
        # Syria
        if (country.name).lower() == 'syrian arab republic':
            country_dictionary[(country.name).lower()] += ['syria']
            
        # Russia
        elif (country.name).lower() == 'russian federation':
            country_dictionary[(country.name).lower()] += ['russia']
        
        # UK
        elif (country.name).lower() == 'united kingdomn':
            country_dictionary[(country.name).lower()] += ['uk', 'great britain']

        # Feed the empty list
        country_set.append((country.alpha_2).lower())
        country_set.append((country.alpha_3).lower())
        country_set.append((country.name).lower())
    
    # Add the extra countries
    country_set += ['syria', 'russia', 'uk', 'great britain']
    
    # Make the list a set
    country_set = set(country_set)
    
    return country_dictionary, country_set
    
    
def country_mentions(country_dictionary, country_set, emails):
    """ This function returns a dictionary that counts the number of email where a country is mentioned.
    It takes as inputs:
    
    @country_dictionary: is the dictionary (key,value):(name_country:[listi other codes])
    @emails: dataframe that contains the data
    """
    
    # Initialize the dictionary
    mentions_dictionary = defaultdict(int)
    
    # Get the manes of all the countries
    keys = list(country_dictionary.keys())
    
    # For each email
    for mail in emails['ExtractedBodyText_2']:
        # Check whether each word is or not in the set of countries
        for i in set(mail):
            
            # Whether there is:
            if i in country_set:  
                
                # In order to identify which country we are talking about
                for k in keys:
                    
                    # Whether in the text we found exactly the name, we add the count to the dictionary
                    if k == i:
                        mentions_dictionary[k] += 1
                    # Otherwise go through the list of other ways to express the country
                    else:                        
                        values = country_dictionary[k]

                        for v in values:
                            # And store it whether found
                            if v == i:
                                mentions_dictionary[k] += 1      
    
    return mentions_dictionary
    
    
def plot_mentions(dict_big_mentions):
    """ This function generate a barplot to show the mentions of countries.
    It takes as input:
    
    @dict_big_mentions: the dictionary to plot"""
    
    
    plt.figure(figsize=(16,6))
    plt.bar(range(len(dict_big_mentions)), list(dict_big_mentions.values()), width=0.7)
    plt.xticks(range(len(dict_big_mentions)), list(dict_big_mentions.keys()))
    plt.xticks(rotation= 60)
    plt.ylabel('No. Mentions')
    plt.xlabel('Countries')
    plt.title('Country mentions')
    plt.show()
    

def demo_liu_hu_lexicon(sentence):
    """ THIS IS JUST BIT MODIFIED 
    Basic example of sentiment classification using Liu and Hu opinion lexicon.
    This function simply counts the number of positive, negative and neutral words
    in the sentence and classifies it depending on which polarity is more represented.
    Words that do not appear in the lexicon are considered as neutral.

    :param sentence: a sentence whose polarity has to be classified.
    :param plot: if True, plot a visual representation of the sentence polarity.
    """
    from nltk.corpus import opinion_lexicon

    pos_words = 0
    neg_words = 0

    y = []

    for word in sentence:
        if word in opinion_lexicon.positive():
            pos_words += 1
            y.append(1) # positive
        elif word in opinion_lexicon.negative():
            neg_words += 1
            y.append(-1) # negative
        else:
            y.append(0) # neutral

    if pos_words > neg_words:
        return 'Positive'
    elif pos_words < neg_words:
        return 'Negative'
    elif pos_words == neg_words:
        return 'Neutral'


def country_mentions_sentiment_vader(country_dictionary, country_set, emails):
    """ This function returns a dictionary that counts the number of email where a country is mentioned.
    It takes as inputs:
    
    @country_dictionary: is the dictionary (key,value):(name_country:[listi other codes])
    @country_set: set of all possible way of representing the countries
    @emails: dataframe that contains the data
    """
    
    # Initialize the dictionary
    mentions_dictionary = defaultdict(int)
    sentiment_dictionary = {}

    for country in list(country_dictionary.keys()):
        sentiment_dictionary[country] = {}
        sentiment_dictionary[country]['pos'] = []
        sentiment_dictionary[country]['neg'] = []
    
    # Train the sentiment analysis
    analyzer = SentimentIntensityAnalyzer()

    # Get the manes of all the countries
    keys = list(country_dictionary.keys())

    # For each email
    for mail in emails['ExtractedBodyText_2']:

        mail_text = ' '.join(mail)

        polarity = analyzer.polarity_scores(mail_text)


        # Check whether each word is or not in the set of countries
        for i in set(mail):

            # Whether there is:
            if i in country_set:  

                # In order to identify which country we are talking about
                for k in keys:

                    # Whether in the text we found exactly the name, we add the count to the dictionary
                    if k == i:
                        mentions_dictionary[k] += 1

                        #sentiment_dictionary[k] = defaultdict(list)

                        sentiment_dictionary[k]['pos'] += [polarity['pos']]
                        sentiment_dictionary[k]['neg'] += [polarity['neg']]

                    # Otherwise go through the list of other ways to express the country
                    else:                        
                        values = country_dictionary[k]

                        for v in values:
                            # And store it whether found
                            if v == i:
                                mentions_dictionary[k] += 1     

                                #sentiment_dictionary[k] =defaultdict(list)

                                sentiment_dictionary[k]['pos'] += [polarity['pos']]
                                sentiment_dictionary[k]['neg'] += [polarity['neg']]

    return mentions_dictionary, sentiment_dictionary


def sentiment_liu(country_dictionary, country_set, emails):
    """ This funtion returns a dictionary that contains for each country (key) a list of the sentiment of the emails
    where the country is mentioned.
    
    It takes as inputs:
    
    @country_dictionary: is the dictionary (key,value):(name_country:[listi other codes])
    @country_set: set of all possible way of representing the countries
    @emails: dataframe that contains the data"""
    
    # Initialize the dictionary
    sentiment_dictionary = {}

    for country in list(country_dictionary.keys()):
        sentiment_dictionary[country] = {}
        sentiment_dictionary[country]['sentiment'] = []

    # Get the manes of all the countries
    keys = list(country_dictionary.keys())
    s= 0
    # For each email
    for mail in emails['ExtractedBodyText_2']:
        # print (s)
        

        polarity = demo_liu_hu_lexicon(mail)


        # Check whether each word is or not in the set of countries
        for i in set(mail):

            # Whether there is:
            if i in country_set:  

                # In order to identify which country we are talking about
                for k in keys:

                    # Whether in the text we found exactly the name, we add the count to the dictionary
                    if k == i:

                        if polarity == 'Positive':
                            sentiment_dictionary[k]['sentiment'] += [1]
                        elif polarity == 'Negative':
                            sentiment_dictionary[k]['sentiment'] += [-1]
                        elif polarity == 'Neutral':
                            sentiment_dictionary[k]['sentiment'] += [0]



                    # Otherwise go through the list of other ways to express the country
                    else:                        
                        values = country_dictionary[k]

                        for v in values:
                            # And store it whether found
                            if v == i:

                                if polarity == 'Positive':
                                    sentiment_dictionary[k]['sentiment'] += [1]
                                elif polarity == 'Negative':
                                    sentiment_dictionary[k]['sentiment'] += [-1]
                                elif polarity == 'Neutral':
                                    sentiment_dictionary[k]['sentiment'] += [0]
      
        s += 1
   
    return sentiment_dictionary         




def plot_sentiment(df_sentiment):
    """ ---------------------------------------------------------
    CREDITS TO: 
    http://stackoverflow.com/questions/31313606/pyplot-matplotlib
    -bar-chart-with-fill-color-depending-on-value 
    ------------------------------------------------------------

    This function plot the sentiment analysis for the country. 
    
    It takes as input:
    
    @df_sentiment"""
    
    # Set up colors : red to green
    plt.figure(figsize=(17,6))
    y = np.array(df_sentiment['sentiment'])
    colors = cm.YlGn(y / float(max(y)))
    plot = plt.scatter(y, y, c=y, cmap = 'YlGn')
    plt.clf()
    clb = plt.colorbar(plot)
    clb.ax.set_title("Sentiment")

    # Display bar plot : country frequency vs. country name, with color indicating polarity score
    plt.bar(range(df_sentiment.shape[0]), df_sentiment['mention'], tick_label=df_sentiment.index, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Country")
    plt.ylabel("Mentions")
    plt.title('Sentiment analysis per country', fontsize = 15)
    plt.show()