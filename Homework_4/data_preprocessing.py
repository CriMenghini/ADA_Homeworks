# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:43:11 2016

@author: cristinamenghini
"""
"""---------------------------------------------------------------------------

    This script stores the functions used to pre-process the data.

---------------------------------------------------------------------------"""

# Required libraries
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import skewtest
from sklearn import preprocessing


def binary_labels(x):
    """ The fuction returns binary labels for a specific entries of an object
    (i.e. array/Series). 
    
    It takes in input:    
    @x : the entry."""
    
    # Whether the entry is smaller or equal than the threshold
    if x <= 0.5:
        #Define it as the label 0
        return 0    
    else:
        # Otherwise as 1
        return 1
        
        
def preprocess_labels(label):
    """ The function encodes the unique values of a variable transforming them 
    into integers. It returns the new labels and takes in input:
    
    @label: array to encode."""
    
    # Apply the encoder
    le = preprocessing.LabelEncoder()
    le.fit(label)
    # Transform the variable
    label = le.transform(label) 
    
    return label
    

def encode_string_variable(df, attribute):
    """ The fuction encodes the attributes of type 'object'. It uses the sklearn function
    LabelEncoder. To each different string is associated an integer. 
    It takes as inputs:
    @df: the dataframe that contain the variable to be processed
    @attribute: the name of the attribute to process (str)"""
    
    # Fill the NaN with 'Unknown'
    df[attribute] = df[attribute].fillna('Unknown')
    
    # Define the encoder for the attribute
    le = preprocessing.LabelEncoder()
    
    # Apply the encoder
    le.fit(df[attribute])
    
    df[attribute] = le.transform(df[attribute]) 

    
    
def create_bins(df, attribute):
    """ This function defines the bins that are going to be used to categorise
    the numerical variables. 
    
    It takes as inputs:
    @df: the dataframe that contain the variable to be processed
    @attribute: is the name of the variable
    
    ---------------------------------------------------------------------------
    
    In particular, it is built on two steps. The first one provide the 
    computation of the skewness of the distribution of the attribute without 
    taking into account those samples that don't lie in the IQR, hence the 
    plausible outliers. 
    
    In general the skewness of an attribute is an in indicator of the simmetry 
    of its distibution. Whether it is a positive value there is more weight in
    left tail of the distribution, otherwise (negative values) the weight is in 
    the right tail. 
    
    The Skew Test is performed to check whether the Skew is significally 
    different from 0. Precisely:
    
    H0: the skew of the distribution the data are drawn from is equal to that
        of the normal distribution (equal to 0).
    
    ---------------------------------------------------------------------------    
    
    The result of the test determines the way in which the bins are created. In
    particular, whether the Skew is significally different from zero,
    the method used to create the bins is the Doane, a particular estimator
    which takes into account the skew of the data. 
    Otherwise the Auto method is used to estimate the bins.    
    
    (a brief description of these estimators is avaiable in this documentation: 
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html).
    
    ------------------------------------------------------------------------"""
        
        
    # Get the end points of the IQR
    B = plt.boxplot(df[attribute])
    plt.close()
    min_max = [item.get_ydata()[1] for item in B['whiskers']]

    # Perform the statistical test
    skew_pvalue = skewtest(df[attribute][df[attribute] >= min_max[0]])[1]
    
    # Whether significally different from zero
    if skew_pvalue < 0.05:
        # Use the Doane method
        bins = np.histogram(df[attribute], bins = 'doane')[1]
        bins_interval = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    # Otherwise
    else:
        # Use the auto method
        bins = np.histogram(df[attribute], bins = 'auto')[1]
        bins_interval = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    
    return bins_interval
    
    
def categorisation(bins_intervals,x):
    """ The function transform the variables according to the bins obtatined 
    through the create_bins function.
    
    It takes as inputs:
    @bins_intervals: is the list of intervals that defines the bins;
    @x: is the entry which the function is applied.
    """
    
    # Define the possible values of the variable    
    classes = range(len(bins_intervals))
    
    # For each bin's interval
    for i in classes:
        # Check whether the entry belongs to it
        if  bins_intervals[i][0] <= x < bins_intervals[i][1]:
            # Reassign the label
            return classes[i]
            
    # Whether the variable is in any interval, it means that is equal to the 
    # right endpoint of the last interval.
    return classes[-1]   
    
    
    
def remove_nans(data_clean, variables_with_nan, players_attributes):
    """ The function returns the input dataframe cleaned from NaNs. 
    It takes as inputs:
    @data_clean: dataframe to process
    @variables_with_nan: dictionary (key:values) key = feature,
                         values = index of nulls
    @players_attributes: list of player's attributes
    
    ----------------------------------------------------------------
    The cleaning procedure provides two steps.
    
    The first check the possibility of imputing the missing values:
    - for 'object' variable by using the first value, since we verify 
      that each player registers only one position
    - for the 'numeircal' we use the median since it is more robust 
      than the mean
    The second drops the dyads whose values have not been imputed.
    --------------------------------------------------------------"""

    for attribute in players_attributes:

        # Get the list of players without attribute's entry
        players_without_attribute = data_clean[variables_with_nan[attribute]]['playerShort'].unique()

        # For each player verify if none dyads contain the information
        for player in players_without_attribute:
            # get number of dyads
            number_dyads = len(data_clean[data_clean['playerShort'] == player][attribute])        
            # Get the index of empty dyads
            index_empty_dyads = data_clean[data_clean['playerShort'] == player][attribute].isnull()
            number_empty_dyads = sum(index_empty_dyads)
            
            # If the number of dyads is different from the number of empty dyads we infer the value
            if number_dyads != number_empty_dyads:
                # Get the index of the not empty dyads
                index_full_dyads = data_clean[data_clean['playerShort'] == player][attribute].notnull()
                data_full = data_clean[index_full_dyads]
                
                # Check which kind of attribute we are dealing with
                if data_clean[attribute].dtypes == 'O':            
                    # Whether it is a strin replace the nulls with the first value
                    replace_value = data_full[attribute][0]
                    data_clean[attribute] = data_clean[attribute].fillna(replace_value)
                
                else:
                    # Whether is a number replace the nulls with the median
                    replace_value = data_full[attribute].median()
                    data_clean[attribute] = data_clean[attribute].fillna(replace_value)

    # Hence we check again for the presence of NaNs in these variable and we proceed dropping the players 
    #whose values have not been imputed
    data_clean = data_clean.dropna(axis=0, subset=['height', 'weight', 'position'])
    
    return data_clean