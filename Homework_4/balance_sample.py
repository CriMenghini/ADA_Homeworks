# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:59:06 2016

@author: cristinamenghini
"""

# Import useful libraries
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


def weight_sample(labels):
    
    weight_class = labels.value_counts()/len(labels)
    
    sample_weights = []
    for i in labels:
        sample_weights += [weight_class[i]]
        
    return np.array(sample_weights)
    
    
def cross_validation(df, labels, estimators, depth):
    
    no_splits = 10
    kf = KFold(n_splits = no_splits, shuffle = True, random_state = 1)

    prediction_roc_train = []
    prediction_fbeta_train = []

    prediction_roc_test = []
    prediction_fbeta_test = []

    for train_index, test_index in kf.split(df):

        X_train = df.iloc[train_index]
        y_train = labels.iloc[train_index]
        X_test = df.iloc[test_index]
        y_test = labels.iloc[test_index]

        train_sample_weights = weight_sample(y_train)

        forest = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=1, class_weight='balanced')
        train_fit = forest.fit(X_train, y_train, sample_weight = train_sample_weights)
        
        prediction_train = train_fit.predict(X_train)
        prediction_test = train_fit.predict(X_test)


        prediction_roc_train += [metrics.roc_auc_score(y_train, prediction_train)]
        prediction_fbeta_train += [metrics.fbeta_score(y_train, prediction_train, beta=1.2)]

        prediction_roc_test += [metrics.roc_auc_score(y_test, prediction_test)]
        prediction_fbeta_test += [metrics.fbeta_score(y_test, prediction_test, beta=1.2)]
        
    return prediction_roc_train, prediction_fbeta_train, prediction_roc_test, prediction_fbeta_test
    

def tuning_cv(players, labels, list_depths = range(3,50, 2), list_numbers_estimators = range(2,100, 5)):
    
    average_roc_train = []
    std_roc_train = []
    average_fbeta_train = []
    std_fbeta_train = []

    average_roc_test = []
    std_roc_test = []
    average_fbeta_test = []
    std_fbeta_test = []

    couples_estimators = []

    for estimator in list_numbers_estimators:
        for depth in list_depths:
            couples_estimators += [(estimator, depth)]
            

            cv = cross_validation(players, labels, estimator, depth)

            average_roc_train += [np.mean(cv[0])]
            std_roc_train += [np.std(cv[0])]
            average_fbeta_train += [np.mean(cv[1])]
            std_fbeta_train += [np.std(cv[1])]


            average_roc_test += [np.mean(cv[2])]
            std_roc_test += [np.std(cv[2])]
            average_fbeta_test += [np.mean(cv[3])]
            std_fbeta_test += [np.std(cv[3])]   
            
    return average_roc_train, std_roc_train, average_fbeta_train, std_fbeta_train, average_roc_test, std_roc_test, average_fbeta_test, std_fbeta_test, couples_estimators