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