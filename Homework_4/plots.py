# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:57:33 2016

@author: cristinamenghini
"""
"""---------------------------------------------------------------------------

    This script stores the functions used to make plots.

---------------------------------------------------------------------------"""


# Import useful libraries
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

"""---------------------------------------------------------------------------------------
						Function to analyse the labels
---------------------------------------------------------------------------------------"""
def boxplot_raters(label_1, label_2): 
	""" This fuction return the boxplot of the distribution of the labels respect the two 
	raters.
	
	It takes as inputs:
	@label_1: the labels given by rater 1
	@label_2: the labels given by rater 2"""
    
    # Array to plot
    y0 = np.array(label_1)
    y1 = np.array(label_2)

	# Define the box-plot related to rater 1
    trace0 = go.Box(name = 'Rater 1',
        			y=y0)
    
    # Define the box-plot  relates to rater 2
    trace1 = go.Box(y=y1,
        			name = 'Rater 2')
    
    # List of data to plot
    data = [trace0, trace1]
    
    # Set up the layout
    layout = go.Layout(title='Distribution of the scores given by the raters',
    				   # Give labels to the axis
                	   xaxis=dict(title='Raters'),
                	   yaxis=dict(title='Scores'))
                	   
    # Define the figure and plot it            	   
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='rater-distr')


    

def stacked_plot(label_1, label_2):
	""" This funtion creates a plot that shows the cumulative percentage of the labels 
	given by the raters.
	
	It takes as inputs:
	@label_1: the array of labels given by rater 1
	@label_2: the array of labels given by rater 2"""
	
	# Cumulative values
    frequ_label_1 = list(np.cumsum(label_1.value_counts())/len(label_1))
    frequ_label_2 = list(np.cumsum(label_2.value_counts())/len(label_2))
    
    
    trace1 = go.Bar(x=['Very light skin', 'Light skin', 'Neither dark nor light skin', 'Dark skin', 'Very dark skin'],
        			y=frequ_label_1,
        			name='Rater 1')
    
    trace2 = go.Bar(x=['Very light skin', 'Light skin', 'Neither dark nor light skin', 'Dark skin', 'Very dark skin'],
        	        y=frequ_label_2,
        			name='Rater 2')

	# Define the data
    data = [trace1, trace2]
    
    
    layout = go.Layout(barmode='stack',
        			   title = 'Cumulative percentage of skin tone classes present in the sample',
         			   yaxis = dict(title='Cumulaative percentage'),
                       xaxis = dict(title='Skin tone'))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='stacked-bar')
    
    
def error_bars(couples_estimators, average_f_score, std_f_score):
	""" This function create the errors bar plot for the average f-scores obtained by the CV.
	
	It takes as inputs:
	@couples_estimators: couple of estimators used for the CV
	@average_f_score: vector of the average f_score
	@std_f_score: vector of std of the f_scor for each step of the CV"""
    
    data = [go.Scatter(x = list(range(len(couples_estimators))),
            		   y = average_f_score,
            		   error_y=dict(type='data',
                	   array = std_f_score,
                       visible=True))]
    
    layout = dict(title = 'Standard deviation of Fbeta scores',
                  yaxis = dict(title='FBeta'),
                  xaxis = dict(title='Parameters combinations'))
                 
    
    fig = go.Figure(data=data, layout=layout)  
    py.iplot(fig, filename='error-bar-asymmetric-array')
    
    
def bubble_plot(labels, players, additional_attr, fun_add, attr_x, fun_x, attr_y, fun_y):
    """ This function draw a bubble plot whose bubbles reppresent the clubs, the diameter
    the proportion of dark skin players, the color the League.
    
    It takes as inputs:
    @labels: Series of labels
    @players: df containing the data
    @additional_attr: additional information about the club (str of the attribute)
    @fun_add: aggregate function for the additional feature
    @attr_x: attribute on the x-axis (str attribute)
    @fun_x: aggregate function for the x-axis feature
    @attr_y:  attribute on the y-axis (str attribute)
    @fun_y: aggregate function for the y-axis feature"""
    
    
    # Define the DF to plot
    # Create the labels attribute
    players['label'] = labels
    # Select the variable of interest
    players_club = players[['leagueCountry', 'club',additional_attr,attr_x, attr_y, 'label']]
    # Sort the samples according to the League and the Club
    players_club = players_club.sort_values(['leagueCountry', 'club'])
    # Aggregate according to the club
    players_club = players_club.groupby('club').agg({'label' : 'mean',
                                                      attr_y : fun_y,
                                                      attr_x : fun_x,
                                                      additional_attr : fun_add,
                                                      'leagueCountry' : 'first',
                                                      'club':'first'})
    
    # Initialize the list of the bubbles size and the features shown in the pop up
    hover_text = []
    bubble_size = []

    # For each club get the informations related to the features of interest
    for index, row in players_club.iterrows():
        hover_text.append(('club: {club}<br>'+
                          'weight: {weight}<br>'+
                          'height: {height}<br>'+
                          'yellowCards: {yellowCards}<br>' +
                           'label: {label}').format(club=row['club'],
                                                weight=row['weight'],
                                                height=row['height'],
                                                yellowCards=row['yellowCards'],
                                                label=row['label']))
        bubble_size.append((row['label'])*100)

    # Add to the df the columns related to the pop-ups and the bubble size
    players_club['text'] = hover_text
    players_club['size'] = bubble_size


    # Define the bubbles related to each League
    trace0 = go.Scatter(x=players_club['weight'][players_club['leagueCountry'] == 'Germany'],
        			    y=players_club['height'][players_club['leagueCountry'] == 'Germany'],
        				mode='markers',
        				name='Germany',
        				text=players_club['text'][players_club['leagueCountry'] == 'Germany'],
       				    marker=dict(symbol='circle',
            						sizemode='diameter',
            						sizeref=0.85,
            						size=players_club['size'][players_club['leagueCountry'] == 'Germany'],
            			line=dict(width=2)))

    trace1 = go.Scatter(x=players_club['weight'][players_club['leagueCountry'] == 'France'],
        			    y=players_club['height'][players_club['leagueCountry'] == 'France'],
        				mode='markers',
        				name='France',
        				text=players_club['text'][players_club['leagueCountry'] == 'France'],
        				marker=dict(symbol='circle',
            					    sizemode='diameter',
             						sizeref=0.85,
            			size=players_club['size'][players_club['leagueCountry'] == 'France'],
            			line=dict(width=2)))

    trace2 = go.Scatter(x=players_club['weight'][players_club['leagueCountry'] == 'England'],
        				y=players_club['height'][players_club['leagueCountry'] == 'England'],
        				mode='markers',
        				name='England',
        				text=players_club['text'][players_club['leagueCountry'] == 'England'],
        				marker=dict(symbol='circle',
            						sizemode='diameter',
            						sizeref=0.85,
            						size=players_club['size'][players_club['leagueCountry'] == 'England'],
            						line=dict(width=2)))

    trace3 = go.Scatter(x=players_club['weight'][players_club['leagueCountry'] == 'Spain'],
        				y=players_club['height'][players_club['leagueCountry'] == 'Spain'],
        				mode='markers',
        				name='Spain',
        				text=players_club['text'][players_club['leagueCountry'] == 'Spain'],
        				marker=dict(symbol='circle',
            						sizemode='diameter',
            						sizeref=0.85,
            						size=players_club['size'][players_club['leagueCountry'] == 'Spain'],
            						line=dict(width=2)))


    data = [trace0, trace1, trace2, trace3]
    
    layout = go.Layout(title='Weight vs. Height',
        			   xaxis=dict(title='Average weight in the club',
            					  gridcolor='rgb(255, 255, 255)',
            					  range=[60, 95],
            					  zerolinewidth=1,
            					  ticklen=5,
                                  gridwidth=2),
        			   yaxis=dict(title='Average height in the club',
            					  gridcolor='rgb(255, 255, 255)',
            					  range=[170, 195],
            					  zerolinewidth=1,
            					  ticklen=5,
            					  gridwidth=2),
        			   paper_bgcolor='rgb(243, 243, 243)',
        			   plot_bgcolor='rgb(243, 243, 243)')

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='players-physics')
    
    
    
    
def boxplot_plotly(players, labels, feature):
    """ This function create the boxplot to represent the distributions of a numerical feature
    in the class 0 and 1.
    
    It takes as inputs:
    @players: df containing the entire data
    @labels: Series of labels
    @feature: str of the feature whose distributions we want to plot"""
    
    
    players_referee = players.copy()
    players_referee['label'] = labels
    
    y0 = players_referee[feature][players_referee['label']==0]
    y1 = players_referee[feature][players_referee['label']==1]

    trace0 = go.Box(y=y0,
        			name = 'Light skin')
    
    trace1 = go.Box(y=y1,
        			name = 'Dark skin')

    layout = go.Layout(title='Distribution of the Average ' + feature + ' on the two classes',
            		   xaxis=dict(title='Classes'),
                       yaxis=dict(title='Average'+ feature))
    
    data = [trace0, trace1]
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=feature)


def scatter_plot(players, labels):
    """ This function represents a scatterplot to see the relation between the meanIAT and
    the number of yellowCards respect to the two classes.
    
    It takes as inputs:
    @players: df containing the entire data
    @labels: Series of labels"""
    
    players_referee = players.copy()
    players_referee['label'] = labels
    
    trace0 = go.Scatter(x = players_referee['meanIAT'][players_referee['label'] == 0],
        			    y = players_referee['yellowCards'][players_referee['label'] == 0],
        				name = 'Light Skin Color',
        				mode = 'markers',
        				marker = dict(size = 8,
            						  color = 'rgba(152, 0, 0, .8)',
            			line = dict(width = 2,color = 'rgb(0, 0, 0)')))

    trace1 = go.Scatter(x = players_referee['meanIAT'][players_referee['label'] == 1],
        				y = players_referee['yellowCards'][players_referee['label'] == 1],
        				name = 'Dark Skin Color',
        				mode = 'markers',
        				marker = dict(size = 8,
            						  color = 'rgba(255,255,0,0.8)',
            			line = dict(width = 2)))

    data = [trace0, trace1]

    layout = dict(title = 'Number of yellow cards vs. meanIAT',
                  yaxis = dict(title='Yellow cards'),
                  xaxis = dict(title='Mean IAT'))

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='styled-scatter-y')

"""---------------------------------------------------------------------------------------
						Function to analyse CV scored
---------------------------------------------------------------------------------------"""    
    
def train_test_plot(couples_estimators, average_roc_train, average_roc_test, average_fbeta_train, average_fbeta_test, plot_name):
	""" This function plot the train and the test AUC, F-beta scores.
	
	It takes as inputs:
	@couples_estimators: couple of estimators used for the CV
	@average_roc_train: vector of average auc in the train cv
	@average_roc_test: vector of average auc in the test cv
	@average_fbeta_train: vector of average f_beta in the train cv
	@average_fbeta_test: vector of average f_beta in the tests cv
	@plot_name: name of the plot (str)"""
    
    random_x = np.array(range(len(couples_estimators)))

    # Create traces
    trace0 = go.Scatter(x = random_x,
        				y = np.array(average_roc_train),
        				mode = 'lines',
        				name = 'Train ROC score')
        				
    trace1 = go.Scatter(x = random_x,
        				y = np.array(average_roc_test),
        				mode = 'lines+markers',
        				name = 'Test ROC score')
    
    trace2 = go.Scatter(x = random_x,
        				y = np.array(average_fbeta_train),
        				mode = 'lines',
        				name = 'Train Fbeta-score')
        				
    trace3 = go.Scatter(x = random_x,
        				y = np.array(average_fbeta_test),
        				mode = 'lines+markers',
        				name = 'Test Fbeta-score')

    data = [trace0, trace1, trace2, trace3]
    
    layout = go.Layout(title='Compare train and test AUC and Fbeta',
             		   xaxis=dict(title='Couple of parameters'),
            		   yaxis=dict(title='AUC, Fbeta'))
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=plot_name)
    
    
def plot_cv_scores(n_fold, cv_scores):
    """ This function plot the cv F scores.
    
    It takes as inputs:
    @n_fold: number of folds
    @cv_scores: list of cv scores"""
    
    folds = ['Fold'+ str(i) for i in range(1,n_fold+1)]

    # Create and style traces
    trace0 = go.Scatter(x = folds,
        				y = cv_scores,
        				name = 'F',
        				line = dict(color = ('rgb(205, 12, 24)'),
            					    width = 4))

    data = [trace0]

    # Edit the layout
    layout = dict(title = 'F score in each folder',
                  xaxis = dict(title = 'Fold'),
                  yaxis = dict(title = 'F score'),
                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='auc-cv')

"""---------------------------------------------------------------------------------------
						Function to analyse the futures importance
---------------------------------------------------------------------------------------"""    
    
def plot_features_importance(players, importances):
    """ This function creates a barplot to represent the feature importance. The most important
    are red. 
    
    It takes as inputs:
    @players: df containing the entire data
    @importances: vector of importances"""
    
    
    trace0 = go.Bar(x=list(players.columns),
    				y=list(importances),
    				marker=dict(color=['rgba(222,45,38,0.8)', 'rgba(222,45,38,0.8)',
               						   'rgba(222,45,38,0.8)', 'rgba(222,45,38,0.8)',
               						   'rgba(222,45,38,0.8)', 'rgba(222,45,38,0.8)', 
               						   'rgba(204,204,204,1)', 'rgba(204,204,204,1)', 
               						   'rgba(204,204,204,1)', 'rgba(204,204,204,1)',
               						   'rgba(204,204,204,1)','rgba(204,204,204,1)',
               						   'rgba(204,204,204,1)','rgba(204,204,204,1)']))

    data = [trace0]
    
    layout = go.Layout(title='Feature importance',
        			   xaxis=dict(title='Features'),
                	   yaxis=dict(title='Importance'))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='color-bar')
    
    
def balance_cv_plot(f1_score_X1, f1_score_X2, f1_score_X3, f1_score_X4, f1_score_X5, n_fold = 10):
    """ This function shows the CV scores for the classifiers applied to the balanced samples.
    
    It takes as inputs:
    @f1_score_X1: cv scores of data set 1
    @f1_score_X2: cv scores of data set 2
    @f1_score_X3: cv scores of data set 3
    @f1_score_X4: cv scores of data set 4
    @f1_score_X5: cv scores of data set 5
    @n_fold: number of folders for the cv"""
    
    folds = ['Fold'+ str(i) for i in range(1,n_fold+1)]

    # Create and style traces
    trace0 = go.Scatter(x = folds,
        				y = f1_score_X1,
        				name = 'F score X1',
        				line = dict(color = ('rgb(205, 12, 24)'),
            					    width = 2))
    
    trace1 = go.Scatter(x = folds,
        				y = f1_score_X2,
        				name = 'F score X2',
        				line = dict(color = ('rgb(22, 96, 167)'),
            					    width = 2,))
    
    trace2 = go.Scatter(x = folds,
        				y = f1_score_X3,
        				name = 'F score X3',
        				line = dict(color = ('rgb(205, 12, 24)'),
            						width = 2))
            						
    trace3 = go.Scatter(x = folds,
        				y = f1_score_X4,
        				name = 'F score X4',
        				line = dict(color = ('rgb(22, 96, 167)'),
            						width = 2))
            						
    trace4 = go.Scatter(x = folds,
        				y = f1_score_X5,
        				name = 'F score X5',
        				line = dict(color = ('rgb(205, 12, 24)'),
            						width = 2))

    data = [trace0, trace1, trace2, trace3, trace4]

    # Edit the layout
    layout = dict(title = 'CV scores',
                  xaxis = dict(title = 'Folds'),
                  yaxis = dict(title = 'F score'))


    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='balance-cv')
    

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
    
    # Set up the characteristics of the plot
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    # Compute the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # Get the averages and std of the train and test sets
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
	# Plot the "confidence bands"
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
                     
    # Plot the scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
	
	# Add the legend
    plt.legend(loc="best")
    
    return plt  