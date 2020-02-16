__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_get_dataset__ = "Read the data given for the kaggle project."

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import pandas as pd # for construct the data

def split_train_valid(data,valid,shuffle=True):
    """
    Split the data in training and validation set
    @param: data: pandas dataframe - data to split
            valid: int - split the data according this percent
            shuffle: bool - whether or not to shuffle the data. Default = True
    """
    # Split
    if ((valid > 0) and (valid<100)):
        print("\t\tSplit data in training set (" + str(100-valid) + "%)" \
               + " and validation set (" + str(valid) + "%)", end="")
        if shuffle:
            print(" after shuffling.")
            validation_set = data.sample(frac=valid/100) #, random_state=seed)
        else:
            print(".")
            l = data.shape[0]
            validation_set = data.iloc[int(l-valid*l/100):,:]
        train_set = data.drop(validation_set.index)
        print("\t\tFinal train set: " + str(train_set.shape[0]))
        print("\t\tFinal validation set: " + str(validation_set.shape[0]))
    # Do not split
    elif valid == 0:
        train_set = data
        validation_set = pd.DataFrame()
    # Error on valid number
    else:
        print("Error: split percents number is not valid. It should an integer" \
                + " between >0 and <100. Given: " + valid)
    return train_set, validation_set

def get_dataset(number,folder="data/",trunc=-1,shuffle=True,valid=33):
    """
    Read data in the format given for the Kaggle challenge and
    return the train, validation and test set.
    The data is in csv format files and under the name Xtri, Ytri, Xtei
    corresponding to the train, label for train and tets set.
    @param: number: string - number of the file to read
            folder: string - path to the csv data files. Default = "data/"
            trunc: int - limitation to the number of data. Default = -1, means 
            keep all data.
            shuffle: bool - whether or not to shuffle the data. Default = True
            valid: int - say how many percents the validation set is
            from the training data set + validation data set. Default = 33 ==>
            training (67%) + validation (33%) = 100%. If 0, then there will be no
            validation set.
    """

    # Get training data set
    train_set = pd.read_csv(folder+"Xtr"+number+".csv",index_col=0)
    if (trunc != -1):
        train_set = train_set.iloc[:trunc,:]
    print("\t\tLoad " + str(train_set.shape[0]) + " training data" + ".")

    # Split training data set and validation data set
    train_set, validation_set = split_train_valid(train_set,valid,shuffle)

    # Get test data set
    test_set = pd.read_csv(folder+"Xte"+number+".csv",index_col=0)
    if (trunc != -1):
        test_set = test_set.iloc[:trunc,:]
    print("\t\tLoad " + str(test_set.shape[0]) + " test data" + ".")

    # Get label and rane it between -1 and 1
    label = pd.read_csv(folder+"Ytr"+number+".csv",index_col=0)*2 -1
    if (trunc != -1):
        label = label.iloc[:trunc,:]
    print("\t\tLoad " + str(label.shape[0]) + " label" + ".")

    return train_set, validation_set, test_set, label