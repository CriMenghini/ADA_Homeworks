# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 21:03:22 2016

@author: cristinamenghini
"""

import os
import glob
import requests
import functools
import pandas as pd
from bs4 import BeautifulSoup

def create_parameter_dict(list_par, html_file):
    """This function returns a dictionary of that form
    
                {'parameter_1' : {'name_1' : value_1, ..., 'name_p1' : value_p1},
                  'parameter_2' : {'name_1' : value_1, ..., 'name_p2' : value_p2},
                   ...........,
                  'parameter_k' : {'name_1' : value_1, ..., 'name_pk' : value_pk}}
                  
        It takes as input:
        @list_par = the list of parameters (includes : Academic unit, Academic period, 
                                                       Pedagogique period, Semester type);
        @html_file = the html to read."""
    
    # Create the empty dictionary
    format_field = {}
    
    # Insert two fixed values
    format_field['ww_i_reportmodel'] = {'reportmodel' : html_file.input(attrs = {'name':"ww_i_reportmodel"})[0]['value']}
    format_field["ww_i_reportModelXsl"] = {i.string : i['value'] for i in html_file.input(attrs = {'name':"ww_i_reportModelXsl"})}
    
    for f in list_par:
        # For each possible value of the parameter we create a disctionary {name : value}
        format_field[f] = {i.string : i['value'] for i in html_file.input(attrs = {'name': f})[0] if i.string != None}
    
    return (format_field)


def compose_url(dic_par, semester, format_ = 'html', department = 'Informatique'):
    '''
        This function returns the entire url for given arguments
        @dic_par is dictionary returned by function create_parameter_dict
        @semester is the name of the semester
        @format_ is the format we use to download the data
        @department is the name of department
    '''
    base_url = 'http://isa.epfl.ch/imoniteur_ISAP/!GEDPUBLICREPORTS.html?'
    
    # html arguments taken from Postman
    ww_x_GPS = 'ww_x_GPS=-1'
    ww_i_reportModel = 'ww_i_reportModel=133685247'
    ww_i_reportModelXsl = 'ww_i_reportModelXsl=' + dic_par['ww_i_reportModelXsl'][format_]
    ww_x_UNITE_ACAD = 'ww_x_UNITE_ACAD=' + dic_par['ww_x_UNITE_ACAD'][department]
    ww_x_PERIODE_PEDAGO = 'ww_x_PERIODE_PEDAGO=' + dic_par['ww_x_PERIODE_PEDAGO'][semester]
    
    list_ = [ww_i_reportModel, ww_i_reportModelXsl, ww_x_UNITE_ACAD, ww_x_PERIODE_PEDAGO]
    
    # We are folding the list from the left with html arguments to concatenate them using conjunction operator
    foldleft = functools.reduce(lambda x, y: x + '&' + y, list_, base_url + ww_x_GPS)
    
    return foldleft


def fetch_files(data_name, urls):
    '''
        This function downloads the data into the directory specified by @data_name
        @data_name is the name of the directory we want the data to be downloaded to
    '''
    
    # Creating an empty directory if it doesn't exist
    try:
        os.mkdir(data_name)
    except:
        pass

    # Changing the permissions in order to avoid problems
    os.chmod(data_name, 0o777)

    # Remove old files from this directory
    files = glob.glob(data_name + '/*')
    for f in files:
        os.remove(f)

    # Downloading the data into the @data_name directory
    k = 0
    for url in urls:
        k += 1
        print ( k, ' request sent.')
        save_data(url, data_name + '/') # funtion defined in separate file
        print ('Data collected from the ', k, ' request.')
        print ('*****'*10)
    print('Done')
    

def fun_rec(indexes, df_init, df_new = []):
    """This is a recursive function that split a given df. It takes as input:
    
    @indexes : the list of indexes where the dataframe is going to be sliced
    @df_init : the dataframe to split
    @df_new : the list where the new frames are going to be added
    
    It returns the list of all the new frames"""
    
    # We recall the function until the list of indexes has length = 1
    if len(indexes) > 1 :
        
        # Define the range of indexes to use in order to split
        rng = (indexes[1]-indexes[0])
        # Split into two parts the df, the head is saved in a list, the tail is used as input for the recalled function
        df_head, df_tail = pd.DataFrame(df_init[:rng].values), df_init[rng:]
        # Append the head
        df_new.append(df_head)
        
        # Recall the function
        return(fun_rec(indexes[1:], df_tail, df_new))
        
    else: 
        return (df_new)
        

def clean_df(df, path, start_year = 2007, finish_year = 2015):
    """This function take as input the dataframe retrieved from IS-Academia. It returns the cleaned df and 
    saves it in a .csv file.
    
    @df : dataframe to clean
    @path : path of the dir where we want to save data
    @start_year : is an integer that represent the year since we want to save data."""
    
    # Get the info related to the df
    info = list(df.ix[[0]][0].values)[0]
    info_split = info.split(',')
    
    # We save only the df related to the years of interest
    if start_year <= int(info_split[1][:5]) <= finish_year:
    
        # Redefine the columns names, in particular they correspond to the 2nd row of the df
        df.columns = df.iloc[1]
        # Create new columns which describe the df
        df['Academic year'] = info_split[1]
        df['Pedagogic period'] = info_split[2][:20]
    
        # Delete the useless columns
        df = df.drop(df.columns[-3], axis = 1)
        # Drop the useless rows
        df = df.drop(df.index[[0,1]])
        
        df = df.drop(df.loc[(df['Civilité'] != 'Monsieur') & (df['Civilité'] != 'Madame')].index)
        
        # Copy the df on a .csv
        df.to_csv(path + info + '.csv', sep = ',')
    
    return df
    
    
def retrieve_data(url):
    """This function takes in input:
    
    @url : the url we are going to do the request
    
    and returns all the frames containing the data of interest (still dirty)"""
    
    # Make the request
    req = requests.get(url)
    html = req.content
    
    # Create a dataframe from the html
    entire_html = pd.read_html(html)[0]
    # Define the indexes on which we are going to split the df
    index_split = list(entire_html.loc[entire_html[0] == 'Civilité'].index - 1) + [len(entire_html)]
    
    # Get the frames
    data_retrived = fun_rec(index_split, entire_html, [])
    
    return (data_retrived)
    

def save_data(u, path):
    """The fuction save the data from all data retrieved from one request. It takes as input
    
    @u : url of interest"""
    
    retrieved_data = retrieve_data(u)
    
    for frames in retrieved_data:
        clean_df(frames, path)