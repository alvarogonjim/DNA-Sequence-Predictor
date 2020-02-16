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
from largeMargin import LargeMargin
from kernel import Kernel
from utils import kernel_train, kernel_predict, score, split_data, write_predictions


print(
    """
------------------------------------------------------------
        DATASET 0
------------------------------------------------------------
"""
)

fname = "0"
dataset = DataPipeline("data/Xtr" + fname + ".csv")

labels = pd.read_csv("data/Ytr" + fname + ".csv")
y = 2.0 * np.array(labels["Bound"]) - 1

test = DataPipeline("data/Xte" + fname + ".csv")



dataset.X = pd.concat([dataset.X, test.X], axis=0, ignore_index=True)
# dataset.X = pd.concat([dataset.X], axis=0, ignore_index=True)


dataset.compute_k_mers(k=9)

dataset.mismatch(k=9, m=1)

# print(dataset.data[0])
# print(len(dataset.data[0]),len(dataset.data[4]),len(dataset.data[24]), \
#         len(dataset.data[14]),len(dataset.data[3]),len(dataset.data[5]),\
#         len(dataset.data[100+43]),len(dataset.data[100+4]),len(dataset.data[100+6]))
# assert(False)

# print(dataset.data)
# print(dataset.data[-1])
# assert(False)

K9 = Kernel(Kernel.mismatch()).gram(dataset.data)

# print(K9)
# assert(False)

dataset.compute_k_mers(k=10)
dataset.mismatch(k=10, m=1)
K10 = Kernel(Kernel.mismatch()).gram(dataset.data)

# print(dataset.data[0])
# print(len(dataset.data[0]),len(dataset.data[4]),len(dataset.data[24]), \
#         len(dataset.data[14]),len(dataset.data[3]),len(dataset.data[5]),\
#         len(dataset.data[100+43]),len(dataset.data[100+4]),len(dataset.data[100+6]))
# assert(False)

# print(K10)
# assert(False)

dataset.compute_k_mers(k=11)
dataset.mismatch(k=11, m=1)
K11 = Kernel(Kernel.mismatch()).gram(dataset.data)


K = K9 + K10 + K11
# K = K9
# np.savetxt("K9.txt",K9)
# np.savetxt("K10.txt",K10)
# np.savetxt("K11.txt",K11)
# np.savetxt("K.txt",K)

# print(K)
# assert(False)

training = [i for i in range(2000)]
testing = [i for i in range(2000, 3000)]
# training = [i for i in range(300)]
# testing = [i for i in range(300, 600)]
# y = y[:300]

# Careful change the lmda
lmda = 0.0001 #0.00000001
# print(K)
# print(K[training])
# print(np.shape(K[training]))
# print(np.shape(K[training][:, training]))
# print(y, len(y))
# assert(False)

alpha = LargeMargin.SVM(K[training][:, training], y, lmda)
# np.savetxt("alpha.txt",alpha)
# print(alpha)
# assert(False)

pred0 = []
for i in tqdm(testing):
    # print("i", i)
    val = 0
    for k, j in enumerate(training):
        # print("k", k, "j", j)
        # print(i, K[i,j])
        val += alpha[k] * K[i, j]
    # assert(False)
    pred0.append(np.sign(val))

print(pred0)
assert(False)
# print(score(pred0, y[1000:2000]))

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

# training = [i for i in range(2000)]
# testing = [i for i in range(2000, 3000)]

# Careful change the lmda
lmda = 0.00000001


alpha = LargeMargin.SVM(K[training][:, training], y, lmda)

pred1 = []
for i in tqdm(testing):
    val = 0
    for k, j in enumerate(training):
        val += alpha[k] * K[i, j]
    pred1.append(np.sign(val))


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