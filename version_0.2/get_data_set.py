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
import os  # for the path
import glob  # for finding files
import pandas as pd  # for construct the data


def read_data(name, title=""):
    """
    Read the csv files in the format given on kaggle
    @param: name: string - path + generic names of the files to read
            title: string - title of the data. Default empty
    """
    files = glob.glob(name + "[0-9].csv")
    data = pd.DataFrame()
    for i in files:
        data = pd.concat([data, pd.read_csv(i, index_col=0)])
    print("Load " + str(data.shape[0]) + " " + title + ".")
    return data  # .iloc[:50,:]


def split_train_valid(data, valid, shuffle=True):
    """
    Split the data in training and validation set
    @param: data: pandas dataframe - data to split
            valid: int - split the data according this percent
            shuffle: bool - whether or not to shuffle the data. Default = True
    """
    # Split
    if (valid > 0) and (valid < 100):
        print(
            "Split data in training set ("
            + str(100 - valid)
            + "%)"
            + " and validation set ("
            + str(valid)
            + "%)",
            end="",
        )
        if shuffle:
            print(" after shuffling.")
            validation_set = data.sample(frac=valid / 100)  # , random_state=seed)
        else:
            print(".")
            l = data.shape[0]
            validation_set = data.iloc[int(l - valid * l / 100) :, :]
        train_set = data.drop(validation_set.index)
        print("Final train set: " + str(train_set.shape[0]))
        print("Final validation set: " + str(validation_set.shape[0]))
    # Do not split
    elif valid == 0:
        validation_set = pd.DataFrame()
    # Error on valid number
    else:
        print(
            "Error: split percents number is not valid. It should an integer"
            + " between >0 and <100. Given: "
            + valid
        )
    return train_set, validation_set


def get_data_set(folder="data/", shuffle=True, valid=33):
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
    train_set = read_data(dir_path + "/" + folder + "Xtr", "training data")

    # Split training data set and validation data set
    train_set, validation_set = split_train_valid(train_set, valid, shuffle)

    # Get test data set
    test_set = read_data(dir_path + "/" + folder + "Xte", "test data")

    # Get label
    label = read_data(dir_path + "/" + folder + "Ytr", "label")

    return train_set, validation_set, test_set, label


############ Main ############
""" If the file is executed separetely """
if __name__ == "__main__":
    get_data_set()
