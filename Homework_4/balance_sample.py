# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:59:06 2016

@author: cristinamenghini
"""

# Import useful libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt    
    
def users_chunks(n, users_list):
    """This function returns the list of chunks that will be used during the multi-threading.
    - n is the number of threads;
    - user_list is the list of user to split."""
    num = float(len(users_list))/n 
    us_lists = [ users_list [i:i + int(num)] for i in range(0, (n-1)*int(num), int(num))]
    us_lists.append(users_list[(n-1)*int(num):])
    return us_lists
    
def create_df(X_class_0, X_class_1, y_train, indexes, n_df=5):
    
    list_df = []
    list_y = []
    
    for i in range(n_df):  
        df_new = pd.concat([X_class_0.iloc[indexes[i]], X_class_1], axis = 0)
        list_df += [df_new]
        list_y += [y_train[df_new.index]]
        
    return list_df[0], list_y[0], list_df[1], list_y[1], list_df[2], list_y[2], list_df[3], list_y[3], list_df[4], list_y[4]
    

def voting_procedure(prediction_df):
    average_model_predictions = []
    for i in prediction_df.columns:
        #print (i)
        occurrences = prediction_df[i].value_counts()
        if len(occurrences) == 1:
            average_model_predictions += [occurrences.index[0]]
        else:
            if np.any(occurrences[occurrences.index == 1] > occurrences[occurrences.index == 0]):
                average_model_predictions += [1]
            else:
                average_model_predictions += [0]
    return average_model_predictions