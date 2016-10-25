# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:49:47 2016

@author: cristinamenghini
"""

import json
import pandas as pd

def parse_topojson(path):
    """ This function returns the sorted list of cantons' ID.
    It take as input:
    
    @path : the path of the topojson file"""
    
    # Read the json file
    with open(path) as json_data:
        topo_json = json.load(json_data)
        
    # Get the list of object canton
    list_cantons = topo_json['objects']['cantons']['geometries']
    
    # Extract the canton's id
    cantons_id = sorted([i['id'] for i in list_cantons])
    
    return cantons_id
    

def create_map_df(col_1, col_2):
    """ This funtion returns the dataframe that will be filled in the map.
    It takes as input:
    
    @col_1 : the list/array that corresponds to the first df's column 
            (Cantons)
    @col_2 :the list/array that corresponds to the second df's column 
            (Total grant)"""
    
    # Define the entry for the df
    df_data = {'Cantons': col_1, 'Total grants': col_2}
    
    # Create the df
    cantons_data = pd.DataFrame(df_data)
    
    return cantons_data