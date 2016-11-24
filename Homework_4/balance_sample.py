# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:59:06 2016

@author: cristinamenghini
"""

"""---------------------------------------------------------------------------

    This script stores the functions used to assess the quality of the model
    and the balance of the sample.

---------------------------------------------------------------------------"""

# Import useful libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier


def weight_sample(labels):
	""" This function returns the weight for each label.
	
	It takes as inputs:
	@labels: the Series of the labels for our dataset"""
    
    # Get the occurrences of each class
    weight_class = labels.value_counts()/len(labels)
    
    # Initialise the list to store the weights assigned to each observation
    sample_weights = []
    for i in labels:
        sample_weights += [weight_class[i]]
        
    return np.array(sample_weights)
    
    
def cross_validation(df, labels, estimators, depth):
    """ The function performs a 10-folds cross validation. 
    It has been implemented from scratch in order to provide different scores both for the
    train and the test sets.
    
    It takes as inputs:
    @df: the df which the cv is performed on
    @labels: Series of labels for our set
    @estimators: the number of estimators that define the Random Forest
    @depth: the maximum depth of the trees
    """
    
    # Define the number of folds
    no_splits = 10
    # Obtain the k-folds randomly
    kf = KFold(n_splits = no_splits, shuffle = True, random_state = 1)

	# Initialise the list to store the ROC-AUC score on the train
    prediction_roc_train = []
    # Initialise the list to store the f-beta score on the train
    prediction_fbeta_train = []

	# Initialise the same lists for the test
    prediction_roc_test = []
    prediction_fbeta_test = []

	# For each defined folds
    for train_index, test_index in kf.split(df):
		
		# Define the train and test X and Y
        X_train = df.iloc[train_index]
        y_train = labels.iloc[train_index]
        X_test = df.iloc[test_index]
        y_test = labels.iloc[test_index]
		
		# Get the weights of the train
        train_sample_weights = weight_sample(y_train)

		# Define the 'balanced' classifier
        forest = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=1, class_weight='balanced')
        # Train it on the K-1 folds
        train_fit = forest.fit(X_train, y_train, sample_weight = train_sample_weights)
        
        # Obtain the prediction for the train and for the test sets
        prediction_train = train_fit.predict(X_train)
        prediction_test = train_fit.predict(X_test)

		# Append the metrics to the initialised lists (train)
        prediction_roc_train += [metrics.roc_auc_score(y_train, prediction_train)]
        prediction_fbeta_train += [metrics.fbeta_score(y_train, prediction_train, beta=1.2)]

		# Append the metrics to the initialised lists (test)
        prediction_roc_test += [metrics.roc_auc_score(y_test, prediction_test)]
        prediction_fbeta_test += [metrics.fbeta_score(y_test, prediction_test, beta=1.2)]
        
    return prediction_roc_train, prediction_fbeta_train, prediction_roc_test, prediction_fbeta_test
    

def tuning_cv(players, labels, list_depths = range(3,50, 2), list_numbers_estimators = range(2,100, 5)):
    """ This function perform a cross validation to all the possible combinations of parameters which
    should be tuned. 
    
    --------------------------------------------------------------------------------------
    It returns:
    @average_roc_train: vector of Roc-Auc metric's average for each performed cv (on the train)
    @std_roc_train: vector of Roc-Auc metric's std for each performed cv (on the train)
    @average_fbeta_train: vector of F-beta metric's average for each performed cv (on the train)
    @std_fbeta_train: vector of F-beta metric's std for each performed cv (on the train)
    @average_roc_test: vector of Roc-Auc metric's average for each performed cv (on the test)
    @std_roc_test: vector of Roc-Auc metric's std for each performed cv (on the test)
    @average_fbeta_test: vector of F-beta metric's average for each performed cv (on the test)
    @std_fbeta_test: vector of F-beta metric's std for each performed cv (on the test)
    @couples_estimators: list of couple of estimators tested
    
    --------------------------------------------------------------------------------------
    It takes as inputs:
    @players: the dataframe of aggregated data
    @labels : Series of labels for our set
    @list_depths: list of depht parameters to try
    @list_numbers_estimators: the list of the number of estimators to try"""
    
    # Initialise the list of the average roc-auc on the train test for each performed 
    # cross validation 
    average_roc_train = []
    # Initialise the list to store the standard deviation of the performances of the 
    # classifier for each cv 
    std_roc_train = []
    
    # The same lists are initialised for the performances on the test set
    average_fbeta_train = []
    std_fbeta_train = []

    average_roc_test = []
    std_roc_test = []
    average_fbeta_test = []
    std_fbeta_test = []

	# Initialise the list to store the couple of parameters evaluated by the cv
    couples_estimators = []

	# Nested loop to try each couple of parameters
    for estimator in list_numbers_estimators:
        for depth in list_depths:
        
        	# Append the couple of parameters
            couples_estimators += [(estimator, depth)]
            
			# Perform the cross-validation
            cv = cross_validation(players, labels, estimator, depth)

			# Compute and store the average metrics for the train
            average_roc_train += [np.mean(cv[0])]
            std_roc_train += [np.std(cv[0])]
            average_fbeta_train += [np.mean(cv[1])]
            std_fbeta_train += [np.std(cv[1])]

			# Compute and store the average metric for the test
            average_roc_test += [np.mean(cv[2])]
            std_roc_test += [np.std(cv[2])]
            average_fbeta_test += [np.mean(cv[3])]
            std_fbeta_test += [np.std(cv[3])]   
            
    return average_roc_train, std_roc_train, average_fbeta_train, std_fbeta_train, average_roc_test, std_roc_test, average_fbeta_test, std_fbeta_test, couples_estimators
  
"""---------------------------------------------------------------------------------------
						Function for balanced classifiers
---------------------------------------------------------------------------------------"""

def players_chunks(n, player_list):
    """ This function returns the list of chunks that will be used by the different 
    classifiers.
    
    It takes as inputs:
    @n is the number of chunks;
    @player_list is the list of players to split."""
    
    # Define the number of elements in each chunk
    num = float(len(player_list))/n 
    
    # List of lists of chunks
    pl_lists = [ player_list [i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
    
    # Add the last chuck 
    pl_lists.append(player_list[(n-1)*int(num):])
    return pl_lists


def create_df(X_class_0, X_class_1, y_train, indexes, n_df=5):
    """ This function defines and returns the new balanced data frame. Even the labels for 
    each new df are returned.
    
    It takes as inputs:
    @X_class_0: the dataframe containng all the observation that belong to the same class (0)
    @X_class_1: the dataframe containng all the observation that belong to the same class (1)
    @y_train: the Series of labels of the entire initial df
    @indexes: the list of lists of indexes of X_class_0 for each new balanced df
    @n_df: number of df to create
    """
    
	# Initialise the lists of the new dfs and of the relative labels
    list_df = []
    list_y = []
    
    for i in range(n_df):
    	# For each chuck of indexes concatenate the X_class_0 df with the entire X_class_1
    	# on the rows.  
        df_new = pd.concat([X_class_0.iloc[indexes[i]], X_class_1], axis = 0)
        
        # Append the df and the labels to the already initialised lists.
        list_df += [df_new]
        list_y += [y_train[df_new.index]]
        
    return list_df[0], list_y[0], list_df[1], list_y[1], list_df[2], list_y[2], list_df[3], list_y[3], list_df[4], list_y[4]
    

def voting_procedure(prediction_df):
	""" This function average the predictions obtained by the different classifiers. In 
	particular, it counts how many model classify the observation as 0 and how many as 1. 
	The procedure assigns to the observation the class with more occurrences. It returns the
	final predictions.
	
	It takes as input:
	@prediction_df: the dataframe that contains the predictions returned by each classifier"""
    
    # Initialise the list of final prediction
    average_model_predictions = []
    
    # For each player to label
    for i in prediction_df.columns:
        # Count how many models classify it as 0 and 1
        occurrences = prediction_df[i].value_counts()
        
        # Whether all classifiers predict the same class
        if len(occurrences) == 1:
        	# Classify the player in the class that registers the occurrences
            average_model_predictions += [occurrences.index[0]]
        # Otherwise
        else:
        	# Assign the class with the highest number of occurrences
            if np.any(occurrences[occurrences.index == 1] > occurrences[occurrences.index == 0]):
                average_model_predictions += [1]
            else:
                average_model_predictions += [0]
                
    return average_model_predictions