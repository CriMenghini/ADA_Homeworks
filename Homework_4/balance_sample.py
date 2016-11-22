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
    kf = KFold(n_splits = no_splits, shuffle = True, random_state = 1)

    prediction_accuracy_train = []
    prediction_precision_train = []
    prediction_f_score_train = []
    prediction_recall_train = []
    prediction_roc_train = []

    prediction_accuracy_test = []
    prediction_precision_test = []
    prediction_f_score_test = []
    prediction_recall_test = []
    prediction_roc_test = []

    for train_index, test_index in kf.split(df):

        X_train = df.iloc[train_index]
        y_train = labels.iloc[train_index]
        X_test = df.iloc[test_index]
        y_test = labels.iloc[test_index]

        train_sample_weights = weight_sample(y_train)
        test_sample_weights = weight_sample(y_test)

        forest = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=1, class_weight='balanced')
        train_fit = forest.fit(X_train, y_train, sample_weight = train_sample_weights)
        
        prediction_train = train_fit.predict(X_train)
        prediction_test = train_fit.predict(X_test)


        prediction_accuracy_train += [metrics.accuracy_score(y_train, prediction_train, sample_weight=train_sample_weights)]
        prediction_precision_train += [metrics.precision_score(y_train, prediction_train, sample_weight=train_sample_weights)] 
        prediction_f_score_train += [metrics.f1_score(y_train, prediction_train, sample_weight=train_sample_weights)] 
        prediction_recall_train += [metrics.recall_score(y_train, prediction_train, sample_weight=train_sample_weights)]  
        prediction_roc_train += [metrics.roc_auc_score(y_train, prediction_train, sample_weight=train_sample_weights)]
        


        prediction_accuracy_test += [metrics.accuracy_score(y_test, prediction_test, sample_weight=test_sample_weights)]
        prediction_precision_test += [metrics.precision_score(y_test, prediction_test, sample_weight=test_sample_weights)] 
        prediction_f_score_test += [metrics.f1_score(y_test, prediction_test, sample_weight=test_sample_weights)] 
        prediction_recall_test += [metrics.recall_score(y_test, prediction_test, sample_weight=test_sample_weights)]  
        prediction_roc_test += [metrics.roc_auc_score(y_test, prediction_test, sample_weight=test_sample_weights)]
        
    return prediction_accuracy_train, prediction_precision_train, prediction_f_score_train, prediction_recall_train, prediction_roc_train,prediction_accuracy_test, prediction_precision_test, prediction_f_score_test, prediction_recall_test,prediction_roc_test