#!/usr/bin/env python

__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org"
__status__ = "Develop"

import numpy as np
import pandas as pd
from tqdm import tqdm


from DataPipeline import DataPipeline
from SVM import SVM
from kernel import Kernel
from Utils import train, predict, write_predictions

# Global variables
PATH_TRAIN_FILE = 'data/Xtr'
PATH_LABEL_FILE = 'data/Ytr'
PATH_TEST_FILE = 'data/Xte'

print(
    """
------------------------------------------------------------
        DATASET 0
------------------------------------------------------------
"""
)
# Read the first dataset
index_file = "0"
# Training
dataset = DataPipeline(PATH_TRAIN_FILE + index_file + ".csv")
# Labels
labels = pd.read_csv(PATH_LABEL_FILE + index_file + ".csv")
y = 2.0 * np.array(labels["Bound"]) - 1
# Test
test = DataPipeline(PATH_TEST_FILE + index_file + ".csv")

#Combine together the train and test for optimize code
dataset.X = pd.concat([dataset.X, test.X], axis=0, ignore_index=True)


print('COMPUTING K-MERS WITH K=9')
dataset.compute_k_mers(k=9)
dataset.mismatch(k=9, m=1)
K9 = Kernel(Kernel.mismatch()).gram(dataset.data)

print('COMPUTING K-MERS WITH K=10')
dataset.compute_k_mers(k=10)
dataset.mismatch(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

print('COMPUTING K-MERS WITH K=11')
dataset.compute_k_mers(k=11)
dataset.mismatch(k=11, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)

# Combine all
K = K9 + K10 + K11

# Get the training and the testing sets
training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

# Careful change the lmda
lambda_param = 0.00000001

# Train the SVM
model = SVM.fit(K[training][:, training], y, lambda_param)

# Get the predictions for the first dataset
predictions0 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += model[k] * K[i, j]
    predictions0.append(np.sign(val))

print(
    """
------------------------------------------------------------
        DATASET 1
------------------------------------------------------------
"""
)

# Read the second dataset
index_file = "1"
# Training
dataset = DataPipeline(PATH_TRAIN_FILE + index_file + ".csv")
# Labels
labels = pd.read_csv(PATH_LABEL_FILE + index_file + ".csv")
y = 2.0 * np.array(labels["Bound"]) - 1
# Test
test = DataPipeline(PATH_TEST_FILE + index_file + ".csv")

#Combine together the train and test for optimize code
dataset.X = pd.concat([dataset.X, test.X], axis=0, ignore_index=True)

print('COMPUTING K-MERS WITH K=9')
dataset.compute_k_mers(k=9)
dataset.mismatch(k=9, m=1)
K9 = Kernel(Kernel.mismatch()).gram(dataset.data)

print('COMPUTING K-MERS WITH K=10')
dataset.compute_k_mers(k=10)
dataset.mismatch(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

print('COMPUTING K-MERS WITH K=11')
dataset.compute_k_mers(k=11)
dataset.mismatch(k=12, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)

# Combine all
K = K9 + K10 + K11

# Get the training and the testing sets
training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]

# Careful change the parameter lambda
lambda_param = 0.00000001

# Train the SVM
model = SVM.fit(K[training][:, training], y, lambda_param)

# Get the predictions for the second dataset
predictions1 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += model[k] * K[i, j]
    predictions1.append(np.sign(val))

print(
    """
------------------------------------------------------------
        DATASET 2
------------------------------------------------------------
"""
)
# Read the third dataset
index_file = "2"
# Training
dataset = DataPipeline(PATH_TRAIN_FILE + index_file + ".csv")
# Labels
labels = pd.read_csv(PATH_LABEL_FILE + index_file + ".csv")
y = 2.0 * np.array(labels["Bound"]) - 1
# Test
test = DataPipeline(PATH_TEST_FILE + index_file + ".csv")


print('COMPUTING K-MERS WITH K=10')
dataset.compute_k_mers(10)
test.kmers = dataset.kmers
dataset.mismatch(10, 0)
test.mismatch(10, 0)

kernel = Kernel(Kernel.sparse_gaussian(7.8))

# Careful change the lmda
lambda_param = 0.00000001

# Train the model
model = train(kernel, dataset.data, y, lambda_param)

# Get the predictions for the last dataset
predictions2 = predict(kernel, model, dataset.data, test.data)

print(
    """
------------------------------------------------------------
        RESULTS
------------------------------------------------------------
"""
)
# write the results combining all the predictions
predictions = predictions0 + predictions1 + predictions2
write_predictions(predictions)
