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
import numpy as np # for arrays tricks
from itertools import combinations, product # for list operations
import pandas as pd # for read the data
import matplotlib.pyplot as plt # for plots
from multiprocessing import Pool # for multiprocess our code
from functools import partial # for create partial object
import tqdm as pb # for progress bar


# Handmade:
''' Get train, validation, test set and label from the given data 
under panda dataframe '''
from get_dataset import get_dataset
''' Create features in train, validation and test set '''
from compute_kmer_feature import compute_kmer_feature
''' Define kernel methods '''
from kernels import *

from largeMargin import LargeMargin
# from kernels import Kernel
from utils import kernel_train, kernel_predict, write_predictions

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

############ Dataset 0 ############
print(style.bold + style.red + "Dataset 0" + style.normal)

# Parameters
file_number = "0" # for the name of the files to read this dataset
truncation = 10 # number of data to read. -1 --> get all data
s = False # whether shuffle the data
v = 0 # split data in training and validation set following this percent
k = 9 # length of the kmer
m = 1 # number of mismatch allowed

# Loading data
print(style.italic + style.blue + "\tBuild datasets" + style.normal)
train_set, validation_set, test_set, label = get_dataset(file_number, \
    folder="data/",trunc=truncation, shuffle=s,valid=v)

print(style.italic + style.blue + "\tCompute kmers: length=" + str(k) + \
    ", mismatch=" + str(m) + style.normal)
train_set, validation_set, test_set = compute_kmer_feature(train_set, \
                        validation_set, test_set, k, m)

print("\t\tCompute Gram matrix" + style.normal)
gram_matrix_0 = gram_matrix(train_set, scalar_product, normalized=False)

# print(gram_matrix_0)
assert(False)

dataset.compute_k_mers(k=10)
dataset.mismatch(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.compute_k_mers(k=11)
dataset.mismatch(k=11, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)


K = K9 + K10 + K11

training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

# Careful change the lmda
lmda = 0.00000001

alpha = LargeMargin.SVM(K[training][:, training], y, lmda)

pred0 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += alpha[k] * K[i, j]
    pred0.append(np.sign(val))


############ Dataset 1 ############
print(
    """
------------------------------------------------------------
        DATASET 1
------------------------------------------------------------
"""
)

fname = "1"
dataset = DataPipeline("data/Xtr" + fname + ".csv")

labels = pd.read_csv("data/Ytr" + fname + ".csv")
y = 2.0 * np.array(labels["Bound"]) - 1

test = DataPipeline("data/Xte" + fname + ".csv")


dataset.X = pd.concat([dataset.X, test.X], axis=0, ignore_index=True)


dataset.compute_k_mers(k=9)
dataset.mismatch(k=9, m=1)
K9 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.compute_k_mers(k=10)
dataset.mismatch(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

dataset.compute_k_mers(k=11)
dataset.mismatch(k=12, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)


K = K9 + K10 + K11

training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

# Careful change the lmda
lmda = 0.00000001


alpha = LargeMargin.SVM(K[training][:, training], y, lmda)

pred1 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += alpha[k] * K[i, j]
    pred1.append(np.sign(val))


############ Dataset 2 ############
print(
    """
------------------------------------------------------------
        DATASET 2
------------------------------------------------------------
"""
)

fname = "2"
dataset = DataPipeline("data/Xtr" + fname + ".csv")

labels = pd.read_csv("data/Ytr" + fname + ".csv")
y = 2.0 * np.array(labels["Bound"]) - 1

test = DataPipeline("data/Xte" + fname + ".csv")


dataset.compute_k_mers(6)
test.kmers = dataset.kmers

dataset.mismatch(6, 0)
test.mismatch(6, 0)

kernel = Kernel(Kernel.sparse_gaussian(7.8))

# Careful change the lmda
lmda = 0.00000001

alpha = kernel_train(kernel, dataset.data, y, lmda)
pred2 = kernel_predict(kernel, alpha, dataset.data, test.data)

############ Submission ############
print(
    """
------------------------------------------------------------
        RESULTS
------------------------------------------------------------
"""
)

out_fname = "Yte.csv"
predictions = pred0 + pred1 + pred2
write_predictions(predictions, out_fname)
