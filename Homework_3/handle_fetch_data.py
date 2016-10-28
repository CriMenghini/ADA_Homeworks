# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:49:47 2016

@author: cristinamenghini
"""

import sys
import json
import simplejson
import pandas as pd
from decimal import Decimal
from googleplaces import GooglePlaces, types, lang


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
    df_data = {'Canton': col_1, 'Total grants': col_2}
    
    # Create the df
    cantons_data = pd.DataFrame(df_data)
    
    return cantons_data


def fetch_data(universities, google_places):
    """ This function fetch data from GoogleMaps. In particular it returns a dictionary structured as follows:
    
                            {University : {Location : x, 'Canton' : y, 'Web site': i}}
        
        and save the data (.json) into the Data dir.
        It takes as input:
        
        @list_universities: the list of universities of interest"""
    
    university_dict = {}
    for uni in universities:
        print (uni)
        # Make the request
        query = google_places.text_search(uni, location = 'Switzerland')
        
        # Parse the retrieved information
        for place in query.places:
            try:
                # Get GeoCoordinates
                geo_loc = place.geo_location
                print (geo_loc)    
            
                place.get_details()
                # Get the canton 
                canton = extract_canton(place.details['address_components'])
                print (canton)         
            
                # Get the web site
                web_site = place.website
                print (web_site)  
                print ('*'*10)
            except:
                print ('error:', sys.exc_info()[0])
        
            # Store elements in the dict
            university_dict[uni] = {'Location': geo_loc, 'Canton' : canton, 'Web site' : web_site}
        
        # Save the file
        with open('Data/university_cantons_info_1.json', 'w') as file:
            file.write(simplejson.dumps(university_dict))
        
    return university_dict

    
def extract_canton(info_list_dict):
    for info in info_list_dict:
        for elem in info['types']:
            if elem == 'administrative_area_level_1':
                return info['short_name']