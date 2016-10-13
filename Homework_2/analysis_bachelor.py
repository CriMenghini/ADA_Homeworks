# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:57:20 2016

@author: cristinamenghini
"""

import pandas as pd
import glob

def import_data(path):
    """This function imports the .csv files and concatenate them on axis 0. 
    It takes in input:
    
    @path : where the .csv are stored"""
    
    # Retrieve all the files in a shoot
    files = glob.glob(path + '*.csv')
    # Create the df concatenating the list of all frames
    df = pd.concat([pd.read_csv(f, sep = ',').drop('Unnamed: 0', axis = 1) for f in files], axis=0)
    
    # Replace the name of a column
    df.rename(columns={'No Sciper': 'No_Sciper'}, inplace=True)
    
    return (df)
    

def extract_students(frame):
    """This function returns the frame related to the students that 
    have an entry for both Bachelor semestre 1 and Bachelor semestre 6. 
    It takes in input:
    
    @frame : the entire df"""
    
    # Group by the identification number of the students
    grouped_data = frame.groupby('No_Sciper')
    
    # Define the semesters we want the students have been enrolled in.
    bachelor = pd.Series([' Bachelor semestre 1',' Bachelor semestre 6'])
    
    # Get the list of the students (we are interested in) No_Sciper
    students = []
    
    # For each group
    for g in grouped_data.groups:
        # We extract the Pedagocic periods 
        period_attended = grouped_data.get_group(g)['Pedagogic period']
        # Wheater both Semester 1 and semester 6 are included we select the student
        if sum(bachelor.isin(period_attended)) == 2:
            students.append(g)
    
    # Thus, we just keep the df related to those students
    df_students = frame.query("No_Sciper in @students")
    
    return (df_students)