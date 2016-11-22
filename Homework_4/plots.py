# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:57:33 2016

@author: cristinamenghini
"""

# Import useful libraries
import pandas as pd
import numpy as np
import seaborn as sns 
import plotly.plotly as py
import plotly.graph_objs as go


def label_boxplot(label_1, label_2):
    
    labels = label_1.append(label_2)
    raters = [int(2) if i > 1418 else int(1) for i in range(len(labels))]
    df_label = pd.DataFrame(np.array([labels,raters]).T, columns=['Label', 'Rater'])
    
    sns.set(style="ticks")

    # Draw a nested boxplot to show the distribution of labels by rater
    sns.boxplot(x="Rater", y="Label", hue="Rater", data=df_label, palette="PRGn")
    sns.despine(offset=10, trim=True)
    

def stacked_plot(label_1, label_2):
    frequ_label_1 = list(np.cumsum(label_1.value_counts())/len(label_1))
    frequ_label_2 = list(np.cumsum(label_2.value_counts())/len(label_2))
    
    trace1 = go.Bar(
        x=['Very light skin', 'Light skin', 'Neither dark nor light skin', 'Dark skin', 'Very dark skin'],
        y=frequ_label_1,
        name='Rater 1')
    trace2 = go.Bar(
        x=['Very light skin', 'Light skin', 'Neither dark nor light skin', 'Dark skin', 'Very dark skin'],
        y=frequ_label_2,
        name='Rater 2'
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='stack'
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='stacked-bar')
    
    
def error_bars(couples_estimators, average_f_score, std_f_score):
    data = [
        go.Scatter(
            x = list(range(len(couples_estimators))),
            y = average_f_score,
            error_y=dict(
                type='data',
                array = std_f_score,
                visible=True
            )
        )
    ]
    py.iplot(data, filename='error-bar-asymmetric-array')
    
    
def bubble_plot(labels, players, additional_attr, fun_add, attr_x, fun_x, attr_y, fun_y):
    
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
    trace0 = go.Scatter(
        x=players_club['weight'][players_club['leagueCountry'] == 'Germany'],
        y=players_club['height'][players_club['leagueCountry'] == 'Germany'],
        mode='markers',
        name='Germany',
        text=players_club['text'][players_club['leagueCountry'] == 'Germany'],
        marker=dict(
            symbol='circle',
            sizemode='diameter',
            sizeref=0.85,
            size=players_club['size'][players_club['leagueCountry'] == 'Germany'],
            line=dict(
                width=2
            ),
        )
    )

    trace1 = go.Scatter(
        x=players_club['weight'][players_club['leagueCountry'] == 'France'],
        y=players_club['height'][players_club['leagueCountry'] == 'France'],
        mode='markers',
        name='France',
        text=players_club['text'][players_club['leagueCountry'] == 'France'],
        marker=dict(
            symbol='circle',
            sizemode='diameter',
            sizeref=0.85,
            size=players_club['size'][players_club['leagueCountry'] == 'France'],
            line=dict(
                width=2
            ),
        )
    )

    trace2 = go.Scatter(
        x=players_club['weight'][players_club['leagueCountry'] == 'England'],
        y=players_club['height'][players_club['leagueCountry'] == 'England'],
        mode='markers',
        name='England',
        text=players_club['text'][players_club['leagueCountry'] == 'England'],
        marker=dict(
            symbol='circle',
            sizemode='diameter',
            sizeref=0.85,
            size=players_club['size'][players_club['leagueCountry'] == 'England'],
            line=dict(
                width=2
            ),
        )
    )

    trace3 = go.Scatter(
        x=players_club['weight'][players_club['leagueCountry'] == 'Spain'],
        y=players_club['height'][players_club['leagueCountry'] == 'Spain'],
        mode='markers',
        name='Spain',
        text=players_club['text'][players_club['leagueCountry'] == 'Spain'],
        marker=dict(
            symbol='circle',
            sizemode='diameter',
            sizeref=0.85,
            size=players_club['size'][players_club['leagueCountry'] == 'Spain'],
            line=dict(
                width=2
            ),
        )
    )


    data = [trace0, trace1, trace2, trace3]
    layout = go.Layout(
        title='Weight vs. Height',
        xaxis=dict(
            title='Average weight in the club',
            gridcolor='rgb(255, 255, 255)',
            range=[60, 95],

            zerolinewidth=1,
            ticklen=5,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Average height in the club',
            gridcolor='rgb(255, 255, 255)',
            range=[170, 195],
            zerolinewidth=1,
            ticklen=5,
            gridwidth=2,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='players-physics')
    
    
    
    
def boxplot_plotly(players, labels, feature):
    
    players_referee = players.copy()
    players_referee['label'] = labels
    
    y0 = players_referee[feature][players_referee['label']==0]
    y1 = players_referee[feature][players_referee['label']==1]

    trace0 = go.Box(
        y=y0,
        name = 'Light skin',
    )
    trace1 = go.Box(
        y=y1,
        name = 'Dark skin'
    )

    layout = go.Layout(
            title='Distribution of the Average ' + feature + ' on the two classes',
            xaxis=dict(
                title='Classes'
            ),
            yaxis=dict(
                title='Average'+ feature
            ),
        )
    data = [trace0, trace1]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=feature)


def scatter_plot(players, labels):
    
    
    players_referee = players.copy()
    players_referee['label'] = labels
    
    trace0 = go.Scatter(
        x = players_referee['meanIAT'][players_referee['label'] == 0],
        y = players_referee['yellowCards'][players_referee['label'] == 0],
        name = 'Light Skin Color',
        mode = 'markers',
        marker = dict(
            size = 8,
            color = 'rgba(152, 0, 0, .8)',
            line = dict(
                width = 2,
                color = 'rgb(0, 0, 0)'
            )
        )
    )

    trace1 = go.Scatter(
        x = players_referee['meanIAT'][players_referee['label'] == 1],
        y = players_referee['yellowCards'][players_referee['label'] == 1],
        name = 'Dark Skin Color',
        mode = 'markers',
        marker = dict(
            size = 8,
            color = 'rgba(255,255,0,0.8)',
            line = dict(
                width = 2,
            )
        )
    )

    data = [trace0, trace1]

    layout = dict(title = 'Number of yellow cards vs. meanIAT',
                  yaxis = dict(
                    title='Yellow cards'
                ),
                  xaxis = dict(
                    title='Mean IAT'
                )
                 )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='styled-scatter-y')