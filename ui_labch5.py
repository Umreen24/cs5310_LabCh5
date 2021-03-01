# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
IMPORTING PACKAGES
"""
import os
import math
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

from pyeeg import bin_power, spectral_entropy
from docx import Document
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

"""
SUBJECT 1 TASKS
"""
"""
Q1 - Load the data in each of the Matlab files into a data frame.
"""
# Setting current directory and getting files
os.chdir('/Users/umreenimam/Documents/Masters/Masters_Classes/CS_5310/chapter05/lab_data/Data/Subject1')
pre_file = 'Pre.mat'
med_file = 'Med.mat'
post_file = 'Post.mat'

# Loading data 
pre_data = loadmat(pre_file)
med_data = loadmat(med_file)
post_data = loadmat(post_file)

# Extracting data
pre_ = pre_data['data']
med_ = med_data['data']
post_ = post_data['data']

# Create dataframes and transpose them 
pre_t = pd.DataFrame(pre_.T)
med_t = pd.DataFrame(med_.T)
post_t = pd.DataFrame(post_.T)

"""
Q2 - Remove zero-fill-in rows from each data frame.
"""
#pre_t = pre_t.loc[(pre_t != 0).any(axis = 1)]
#med_t = med_t.loc[(med_t != 0).any(axis = 1)]
#post_t = post_t.loc[(post_t != 0).any(axis = 1)]

pre_t = pre_t[~np.all(pre_t == 0, axis = 1)]
med_t = med_t[~np.all(med_t == 0, axis = 1)]
post_t = post_t[~np.all(post_t == 0, axis = 1)]

"""
Q3 - Compute alpha PSIs and spectral entropies.
"""
# Creating function to calculate psis and spectral entropies
def get_psi_entropies(data):
    
    # Setting variables needed for for loop
    band = [0.5, 4, 7, 12, 30, 100]
    fs = 1024
    size = 1024
    
    columns = data.shape[1]
    rows = math.floor(data.shape[0] / size)
    alpha_psi_df = pd.DataFrame([])
    temp_data = pd.DataFrame([])
    
    # Creating for loop
    for x in range(columns):
        alpha_psis = []
        entropies = []
        temp_col_alpha = 'alpha_psi ' + str(x)
        temp_col_entropy = 'spectral_entropy ' + str(x)
        
        for y in range(rows):
            psis, power_ratios = bin_power(data.iloc[(y * size):((y + 1) * size), x], band, fs)
            alpha_psis.append(psis[2])
    
            spec_entropies = spectral_entropy(data.iloc[(y * size):((y + 1) * size), x], band, fs, power_ratios)
            entropies.append(spec_entropies)
        
        temp_data[temp_col_alpha] = alpha_psis
        temp_data[temp_col_entropy] = entropies
        
    alpha_psi_df = alpha_psi_df.append(temp_data, ignore_index = True)
    
    return alpha_psi_df

# Calling alpha psi function on each dataframe
pre_df = get_psi_entropies(pre_t)
med_df = get_psi_entropies(med_t)
post_df = get_psi_entropies(post_t)

"""
Q4 & Q5 - Create list of brain state labels & combine dataframes into one.
"""
pre_label = ["Pre"] * pre_df.shape[0]
med_label = ["Med"] * med_df.shape[0]
post_label = ["Post"] * post_df.shape[0]

frames = [pre_df, med_df, post_df]
sub1_df = pd.concat(frames, ignore_index = True)
state_labels = pre_label + med_label + post_label

"""
Q6-Q8 - Create a correlation coefficient matrix of 68 channels.
"""
