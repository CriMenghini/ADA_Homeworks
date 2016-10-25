# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:01:48 2016

@author: cristinamenghini
"""

import folium

def overlay_map(df, color, topo_path):
    """ This returns as output the map of swiss with cantons and
    saves it as html in a file.
    It takes as inputs:
    
    @df : data to be entered (col_1 should be the cantons and col_2 the
          grants)
    @color : colors you want to use (string)
    @topo_path : is the path of the topojson file
    """
    
    # Define the map of Switzerland
    switzerland_map = folium.Map(location=[46.912457, 8.191891],
                    zoom_start=7)
    
    # Overlay the map
    switzerland_map.choropleth(geo_path = topo_path, 
                           data = df,
                           columns = ['Canton', 'Total grants'],
                           key_on = 'feature.id',
                           fill_color = color,
                           fill_opacity = 0.7,
                           line_opacity = 0.2,
                           legend_name = 'Total grants',
                           topojson = 'objects.cantons')
    
    switzerland_map.save('Viz_map_html_form/overlaid_map.html')