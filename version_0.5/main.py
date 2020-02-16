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
        the data, create the features, "

############ Imports ############
"""
In the context of the project, the goal is to gain practical experience
with machine learning and learn how to implement for solve simple problems.
Thus, the rules say that external machine learning libraries are forbidden, and
so we limit ourself to use only common libraries in python as following.
We have the right to work with some linear algebra and optimization libraries.
"""

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


############ Dataset 0 ############
print(style.bold + style.red + "Dataset 0" + style.normal)

# Parameters
file_number = "0" # for the name of the files to read this dataset
truncation = -1 # number of data to read. -1 --> get all data
s = True # whether shuffle the data
v = 0 # split data in training and validation set following this percent
k = 9 # length of the kmer start
m = 1 # number of mismatch allowed
N = 3 # number of gram matrix to compute
# Initialization of the gram matrix
train_set_gm = 0; validation_set_gm = 0; test_set_gm = 0
gm = 0
l = 0.0001 # regularization parameter

# Loading data
print(style.italic + style.blue + "\tBuild datasets" + style.normal)
train_set, validation_set, test_set, label = get_dataset(file_number, \
    folder="data/",trunc=truncation, shuffle=s,valid=v)
label_training = np.array(label[label.index.isin(train_set.index)])
label_validation = np.array(label[label.index.isin(validation_set.index)])

for step in range(N):
    print(style.italic + style.blue + "\tCompute kmers: length=" + str(k+step) \
        + ", mismatch=" + str(m) + style.normal)
    train_set_kmer, validation_set_kmer, test_set_kmer = \
        compute_kmer_feature(train_set, validation_set, test_set, k+step, m)
    
    # print(train_set_kmer[0])
    # print(len(train_set_kmer[0]),len(train_set_kmer[4]),len(train_set_kmer[24]), \
    #     len(train_set_kmer[14]),len(train_set_kmer[3]),len(train_set_kmer[5]),\
    #     len(test_set_kmer[43]),len(test_set_kmer[4]),len(test_set_kmer[6]))
    # assert(step < 1)

    print("\t\tComputing Gram matrix: ", end="")
    gm += gram_matrix(train_set_kmer, validation_set_kmer, \
        test_set_kmer, scalar_product)
    # np.savetxt(str(step)+".txt",oui)
    print("\r\t\tGram matrix computed.       ")
    # print(gm)
    # assert(step < 1)

# np.savetxt("gm.txt",gm)
# print(gm)
# assert(False)

# Reshape data
# assert False, gm.shape
train_set_gm = gm[:train_set.shape[0],:train_set.shape[0]]
validation_set_gm = gm[train_set.shape[0]:-test_set.shape[0],:train_set.shape[0]]
test_set_gm = gm[-test_set.shape[0]:,:train_set.shape[0]]
label_training = np.squeeze(label_training)
label_validation = np.squeeze(label_validation)

# print(gm)
# print(train_set_gm)
# assert(False)
# assert False, test_set_gm[-1,-1]

# Train SVM model
print(style.italic + style.blue + "\tTrain SVM model" + style.normal)
print("\t\tLambda = " + str(l))
model0 = SVM(l)
alpha0 = model0.fit(train_set_gm, label_training)

# np.savetxt("alpha0.txt",alpha0)
# print(alpha0)
# assert(False)

if  not validation_set.empty:
    # Compute the performances on the validation set
    print("\t\tCompute the performances on the validation set")
    predictions_valid = model0.predict(validation_set_gm)

    print("\t\tScore: "+ \
        str(round(score(predictions_valid, label_validation),3)*100) + "%")

# Predict on test set
print(style.italic + style.blue + "\tPredict on test set" + style.normal)
predictions_0 = model0.predict(test_set_gm)

print(predictions_0)
assert(False)

############ Dataset 1 ############
print(style.bold + style.red + "Dataset 1" + style.normal)

# Parameters
file_number = "1" # for the name of the files to read this dataset
truncation = -1 # number of data to read. -1 --> get all data
s = False # whether shuffle the data
v = 0 # split data in training and validation set following this percent
k = 9 # length of the kmer start
m = 1 # number of mismatch allowed
N = 3 # number of gram matrix to compute
# Initialization of the gram matrix
train_set_gm = 0; validation_set_gm = 0; test_set_gm = 0
gm = 0
l = 0.0001 # regularization parameter

# Loading data
print(style.italic + style.blue + "\tBuild datasets" + style.normal)
train_set, validation_set, test_set, label = get_dataset(file_number, \
    folder="data/",trunc=truncation, shuffle=s,valid=v)
label_training = np.array(label[label.index.isin(train_set.index)])
label_validation = np.array(label[label.index.isin(validation_set.index)])

for step in range(N):
    print(style.italic + style.blue + "\tCompute kmers: length=" + str(k+step) \
        + ", mismatch=" + str(m) + style.normal)
    train_set_kmer, validation_set_kmer, test_set_kmer = \
        compute_kmer_feature(train_set, validation_set, test_set, k+step, m)

    print("\t\tComputing Gram matrix: ", end="")
    gm += gram_matrix(train_set_kmer, validation_set_kmer, \
        test_set_kmer, scalar_product)
    print("\r\t\tGram matrix computed.       ")


# Reshape data
train_set_gm = gm[:train_set.shape[0]][:,:train_set.shape[0]]
validation_set_gm = gm[-validation_set.shape[0]:,:train_set.shape[0]]
test_set_gm = gm[-test_set.shape[0]:,:train_set.shape[0]]
label_training = np.squeeze(label_training)
label_validation = np.squeeze(label_validation)

# Train SVM model
print(style.italic + style.blue + "\tTrain SVM model" + style.normal)
print("\t\tLambda = " + str(l))
model1 = SVM(l)
alpha1 = model1.fit(train_set_gm, label_training)

if  not validation_set.empty:
    # Compute the performances on the validation set
    print("\t\tCompute the performances on the validation set")
    predictions_valid = model1.predict(validation_set_gm)

    print("\t\tScore: "+ \
        str(round(score(predictions_valid, label_validation),3)*100) + "%")

# Predict on test set 1
print(style.italic + style.blue + "\tPredict on test set" + style.normal)
predictions_1 = model1.predict(test_set_gm)


############ Dataset 2 ############
print(style.bold + style.red + "Dataset 2" + style.normal)

# Parameters
file_number = "2" # for the name of the files to read this dataset
truncation = -1 # number of data to read. -1 --> get all data
s = False # whether shuffle the data
v = 0 # split data in training and validation set following this percent
k = 6 # length of the kmer start
m = 1 # number of mismatch allowed
N = 1 # number of gram matrix to compute
# Initialization of the gram matrix
train_set_gm = 0; validation_set_gm = 0; test_set_gm = 0
gm = 0
l = 0.00000001 # regularization parameter

# Loading data
print(style.italic + style.blue + "\tBuild datasets" + style.normal)
train_set, validation_set, test_set, label = get_dataset(file_number, \
    folder="data/",trunc=truncation, shuffle=s,valid=v)
label_training = np.array(label[label.index.isin(train_set.index)])
label_validation = np.array(label[label.index.isin(validation_set.index)])

for step in range(N):
    print(style.italic + style.blue + "\tCompute kmers: length=" + str(k+step) \
        + ", mismatch=" + str(m) + style.normal)
    train_set_kmer, validation_set_kmer, test_set_kmer = \
        compute_kmer_feature(train_set, validation_set, test_set, k+step, m)

    print("\t\tComputing Gram matrix: ", end="")
    gm += gram_matrix(train_set_kmer, validation_set_kmer, \
        test_set_kmer, sparse_gaussian)
    print("\r\t\tGram matrix computed.       ")


# Reshape data
train_set_gm = gm[:train_set.shape[0]][:,:train_set.shape[0]]
validation_set_gm = gm[train_set.shape[0]:-test_set.shape[0],:train_set.shape[0]]
test_set_gm = gm[-test_set.shape[0]:,:train_set.shape[0]]
label_training = np.squeeze(label_training)
label_validation = np.squeeze(label_validation)

# Train SVM model
print(style.italic + style.blue + "\tTrain SVM model" + style.normal)
print("\t\tLambda = " + str(l))
model2 = SVM(l)
alpha2 = model2.fit(train_set_gm, label_training)

if  not validation_set.empty:
    # Compute the performances on the validation set
    print("\t\tCompute the performances on the validation set")
    predictions_valid = model2.predict(validation_set_gm)

    print("\t\tScore: "+ \
        str(round(score(predictions_valid, label_validation),3)*100) + "%")

# Predict on test set
print(style.italic + style.blue + "\tPredict on test set" + style.normal)
predictions_2 = model2.predict(test_set_gm)



############ Final results ############
print(style.bold + style.red + "Final results" + style.normal)
final_predicitons = predictions_0 + predictions_1 + predictions_2
final_predicitons = pd.DataFrame({"Bound": final_predicitons})
final_predicitons = (final_predicitons+1)/2
final_predicitons.index.name = "Id"
title = "Yte_GONZALEZ_LAURENDEAU_kaggle_submission.csv"
final_predicitons.to_csv(title, \
    sep=",", encoding="utf-8", index=True)
print(style.italic + style.blue + "\tPrediction saved under " \
    + title + style.normal)