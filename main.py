__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_main__ = "Kaggle data challenge for the course 'Advanced Learning Models' \
        for the master programs MSIAM and MoSIG. Main file which allows to read \
        the data, create the features, train the model, predict and save the \
        results."

############ Imports ############
"""
In the context of the project, the goal is to gain practical experience
with machine learning and learn how to implement for solve simple problems.
Thus, the rules say that external machine learning libraries are forbidden, and
so we limit ourself to use only common libraries in python as following.
We have the right to work with some linear algebra and optimization libraries.
"""

from os import path

# Pre-installed library: (python3 -m pip install ...)
import numpy as np # for arrays operations
from itertools import combinations, product # for list operations
import pandas as pd # for manipulate the dataframe
from cvxopt import solvers, matrix, spmatrix, sparse # convex optimization


# Handmade:
''' Get train, validation, test set and label from the given data
under panda dataframe '''
from get_dataset import get_dataset
''' Create features in train, validation and test set '''
from compute_kmer_feature import compute_kmer_feature
''' Define kernel methods '''
from kernels import gram_matrix, scalar_product, sparse_gaussian
''' Build SVM model '''
from svm import SVM, score

# Fancy print in the console
class style:
    red = '\033[91m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    bold = '\033[1m'
    italic = '\033[3m'
    mono = '\033[2m'
    underline = '\033[4m'
    normal = '\033[0m'

# ############ Console information ############
print(style.mono + "The project aim to predict whether a DNA sequencies " \
    + "a binding site to a specific transcription factor. \nIn the context of " \
    + "this competition, the goal is to gain practical experience with machine " \
    + "learning and so no external libraries are allowed unless classical python " \
    + "toolkit and some optimization packages. We have used the following one, " \
    + "which are needed to run the file: " + style.normal)
f = open("requirements.txt", "r")
packages = f.readlines()[1:]
for p in packages:
    print("|"+p[:-1]+"|", end="")
print(style.mono + "\nThe study has three different datasets which we treat " \
    + "independently. For each of them, we map the training, validation and test " \
    + "set on kmer feature, then compute a gram matrix and finally use a SVM " \
    + "model to predict the solution. \nThe steps of the run are display on the " \
    + "console, as well as the specific parameters for each datasets.\n" \
    + style.underline + __author__ + " | " + __date__ + style.normal)

############ Main functions ############
def predict_dataset(dataset_numbers, kmer_length, mismatch, lambdas, \
    truncation = -1, shuffle = False, validation = 24):
    """
    Function to predict the label on the datasets with different 
    hyperparameters. 
    @param: dataset_numbers: numpy array of int - number of the datasets {0,1,2}
            kmer_length: numpy array of numpy array of int - length of the 
            different kmers to use for one dataset
            mismatch: numpy array of numpy array of int - number of mismatchs 
            allowed for one dataset
            lambdas: numpy array of int - regularization parameter
            truncation: int - number of data to use. -1 --> use all data
            shuffle: bool - whether shuffle the data. Default = True
            validation: int - = split the data in training and validation set 
            following this percent. Default = 24
    """

    # Prepare return
    predictions = []

    for pred_i in range(len(dataset_numbers)):

        current_dataset_number = str(dataset_numbers[pred_i])

        print(style.bold + style.red + "Dataset " + \
            current_dataset_number + style.normal)

        # Get hyperparameter
        k = kmer_length[pred_i] # the different length of kmer
        m = mismatch[pred_i] # number of the different mismatch allowed
        l = lambdas[pred_i] # regularization parameter

        # Initialization of the gram matrix
        gm = 0

        # Loading data
        print(style.italic + style.blue + "\tBuild datasets" + style.normal)
        train_set, validation_set, test_set, label = get_dataset(current_dataset_number, \
            folder="data/",trunc=truncation, shuffle=shuffle,valid=validation)
        label_training = np.array(label.iloc[ \
            pd.Index(label.index).get_indexer(train_set.index)])
        label_validation = np.array(label.iloc[ \
            pd.Index(label.index).get_indexer(validation_set.index)])

        for step in range(len(k)):

            # Check if the gram matrix was already computed
            file_name = "precomputed"+current_dataset_number+"/" \
                +str(validation)+"_"+str(k[step])+"_"+str(m[step])

            print(style.italic + style.blue + "\tCompute kmers: length=" \
                + str(k[step]) + ", mismatch=" + str(m[step]) + style.normal)

            if not path.exists(file_name):
                train_set_kmer, validation_set_kmer, test_set_kmer = \
                    compute_kmer_feature(train_set, validation_set, test_set, \
                        k[step], m[step])

                print("\t\tComputing Gram matrix: ", end="")
                actual_gm = gram_matrix(train_set_kmer, validation_set_kmer, \
                    test_set_kmer, scalar_product)
                print("\r\t\tGram matrix computed.       ")

                np.savetxt(file_name,actual_gm)
            
            else:
                actual_gm = np.loadtxt(file_name)
                print("\r\t\tGram matrix loaded.")

            gm += actual_gm


        # Reshape data
        train_set_gm = gm[:train_set.shape[0],:train_set.shape[0]]
        validation_set_gm = gm[train_set.shape[0]:-test_set.shape[0],:train_set.shape[0]]
        test_set_gm = gm[-test_set.shape[0]:,:train_set.shape[0]]
        label_training = np.squeeze(label_training)
        label_validation = np.squeeze(label_validation)

        # Train SVM model
        print(style.italic + style.blue + "\tTrain SVM model" + style.normal)
        print("\t\tLambda = " + str(l))
        model = SVM(l)
        alpha = model.fit(train_set_gm, label_training)

        if  not validation_set.empty:
            # Compute the performances on the validation set
            print("\t\tCompute the performances on the validation set")
            predictions_valid = model.predict(validation_set_gm)

            # Score
            sc = str(round(score(predictions_valid, label_validation),3)*100)
            print("\t\tScore: "+ sc + "%")
            f = open("score.txt", "a+")
            f.write("SCORE = " + sc)
            f.write(" $ Dataset: " + current_dataset_number + " - Validation = " \
                + str(validation) + " |")
            for i in range(len(k)):
                f.write(" k " + str(k[i]) + ", m " + str(m[i]))
            f.write(" | lambda = " + str(l) + "\n")
            f.close()

        # Predict on test set
        print(style.italic + style.blue + "\tPredict on test set" + style.normal)
        predictions += [model.predict(test_set_gm)]

    return predictions


def save_result(predictions, name):
    """
    Save the predictions under the Kaggle format scheme
    @param: predictions: list of {0, 1} - predictions done
            name: string - name of the file to save
    """
    print(style.bold + style.red + "Final results" + style.normal)
    final_predictions = predictions
    final_predictions = pd.DataFrame({"Bound": final_predictions})
    final_predictions = (final_predictions+1)/2
    final_predictions.index.name = "Id"
    final_predictions.to_csv(name, sep=",", encoding="utf-8", index=True)
    print(style.italic + style.blue + "\tPrediction saved under " \
        + name + style.normal)

############ Main ############
''' Make some benchmark '''
if __name__ == "__main__":
    ############ Predictions ############
    dataset_numbers = [0, 1, 2]
    kmer_length = [[10], [10], [10, 11]]
    mismatch = [[1], [1], [1, 1]]
    lambdas = [0.4, 0.45, 0.3]
    # dataset_numbers = [0]
    # kmer_length = [[6]]
    # mismatch = [[1]]
    # lambdas = [0.000001]
    predictions = predict_dataset(dataset_numbers, kmer_length, mismatch, lambdas, \
        validation=0, shuffle=True)

    ############ Save results ############
    submit = predictions[0] + predictions[1] + predictions[2]
    save_result(submit, "Yte_GONZALEZ_LAURENDEAU_kaggle_submission.csv")