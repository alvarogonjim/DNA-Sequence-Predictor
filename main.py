"""
@authors: GONZALEZ Alvaro: alvarogonjim95@gmail.com
        & LAURENDEAU Matthieu: laurendeau.matthieu@gmail.com
@date: 2020 January
@brief: Kaggle data challenge for the course "Advanced Learning Models"
        for the master programs MSIAM and MoSIG.
        Main file which allows to
"""

############ Imports ############
"""
In the context of the project, the goal is to gain practical experience 
with machine learning and learn how to implement for solve simple problems.
Thus, the rules say that external machine learning libraries are forbidden, and 
so we limit ourself to use only common libraries in python as following.
We have the right to work with some linear algebra and optimization libraries.
"""
import io; import sys # for write output in log file
out_console = io.StringIO()
f = open("log.txt", "w")
import numpy as np # for arrays tricks
import os ; import glob; import pandas as pd # for read the data
import matplotlib.pyplot as plt # for plots

############ Get data set ############
'''get train, validation, test set and label from given data 
under panda dataframe'''
from get_data_set import get_data_set

sys.stdout = out_console; print("****** LOADING DATA ******")
train_set, validation_set, test_set, label = get_data_set()
print("****** DATA LOADED ******\n"); sys.stdout = sys.__stdout__

print("Train set (" + str(train_set.shape[0]) + "), validation set (" \
        + str(validation_set.shape[0]) + "), test set (" + str(test_set.shape[0]) \
        + ") and label (" + str(label.shape[0]) + ") loaded.")


############ Create features ############
'''create features in train, validation and test set'''
from create_k_mer_features import create_k_mer_features

#TO UNCOMMENT
# sys.stdout = out_console
#TO UNCOMMENT
sys.stdout = out_console; print("****** CREATE FEATURES ******")
train_set, validation_set, test_set = create_k_mer_features()
print("****** FEATURES CREATED ******\n"); sys.stdout = sys.__stdout__
#TO UNCOMMENT
# sys.stdout = sys.__stdout__
#TO UNCOMMENT





f.write(out_console.getvalue())
f.close()