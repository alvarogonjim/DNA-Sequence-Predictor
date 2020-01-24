"""
@authors: GONZALEZ Alvaro: alvarogonjim95@gmail.com
        & LAURENDEAU Matthieu: laurendeau.matthieu@gmail.com
@date: 2020 January
@brief: Read the data given for the kaggle project.
"""

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import numpy as np
import os 
import glob
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# def split_train_valid():

def read_data(folder="data/",shuffle=True,valid=33):
    """
    Read data in the format given for the Kaggle challenge and
    return the train, validation and test set.
    The data is in csv format files and under the name Xtri, Ytri, Xtei
    corresponding to the train, label for train and tets set.
    @param: folder: string - path to the csv data files. Default = "data/"
            seed: 
            shuffle: bool - whether or not to shuffle the data. Default = True
            valid: int - say how many percents the validation set is
            from the training data set + validation data set. Default = 33 ==>
            training (67%) + validation (33%) = 100%. If 0, then there will be no
            validation set.
    """
    # Get path of the directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Get training data set
    xtrfiles = glob.glob(dir_path+"/"+folder+"Xtr[0-9].csv")
    train_set = pd.DataFrame()
    for i in xtrfiles:
        train_set = pd.concat([train_set, pd.read_csv(i,index_col=0)])
    # print("Load " + str(train_set['Id'].count()) + " training data.")

    # print(train_set.count(['seq']))

    # Split training data set and validation data set
    if ((valid > 0) and (valid<100)):
        print("Split data in training set (" + str(100-valid) + "%)" \
               + " and validation set (" + str(valid) + "%)", end="")
        if shuffle:
            print(" after shuffling.")
            # train_set_copy = train_set.copy()
            valid_set = train_set.sample(frac=valid/100) #, random_state=seed)
            # print(str(valid_set['Id'].count()))
            
            # print(str(train_set['Id'].count()))
            train_set.drop(valid_set.index, inplace=True)
            # print(train_set.head())
            # train_set = train_set.drop([2])
            # print(train_set)
            # print(np.sort(valid_set.index))
            # print(train_set['Id'].count())
            # print(train_set.index)
            # valid_mask = np.random.rand(len(train_set)) < valid / 100
            # valid_set = train_set[valid_mask]
            # train_set = train_set[~valid_mask]
        else:
            print(".")
            pass
        
    # print(str(train_set['Id'].count()))


############ Main ############
''' If the file is executed separetely '''
if __name__ == "__main__":
    read_data()