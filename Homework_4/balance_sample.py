# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:59:06 2016

@author: cristinamenghini
"""

# Import useful libraries
import numpy as np
import pandas as pd
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
    kf = KFold(n_splits = no_splits, shuffle = True)

    prediction_accuracy = []
    prediction_precision = []
    prediction_f_score = []
    prediction_recall = []

    for train_index, test_index in kf.split(df):

        X_train = df.iloc[train_index]
        y_train = labels.iloc[train_index]
        X_test = df.iloc[test_index]
        y_test = labels.iloc[test_index]

        train_sample_weights = weight_sample(y_train)
        test_sample_weights = weight_sample(y_test)

        forest = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=1, class_weight='balanced')
        train_fit = forest.fit(X_train, y_train, sample_weight = train_sample_weights)
        prediction = train_fit.predict(X_test)

        prediction_accuracy = [metrics.accuracy_score(y_test, prediction, sample_weight=test_sample_weights)] + prediction_accuracy
        prediction_precision = [metrics.precision_score(y_test, prediction, sample_weight=test_sample_weights)] + prediction_precision
        prediction_f_score = [metrics.f1_score(y_test, prediction, sample_weight=test_sample_weights)] + prediction_f_score
        prediction_recall = [metrics.recall_score(y_test, prediction, sample_weight=test_sample_weights)] + prediction_recall 
        
    return prediction_accuracy, prediction_precision, prediction_f_score, prediction_recall