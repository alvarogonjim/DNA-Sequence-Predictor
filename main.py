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
import numpy as np # for arrays tricks
import os ; import glob; import pandas as pd # for read the data
import matplotlib.pyplot as plt # for plots

############ Get data set ############
'''get train, validation and test set from given data'''
from get_data_set import read_data

read_data()
