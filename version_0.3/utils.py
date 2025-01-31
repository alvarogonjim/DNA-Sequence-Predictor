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
from largeMargin import LargeMargin
from tqdm import tqdm


def write_predictions(predictions, out_fname):

    data = [[int(np.abs((pred + 1) // 2))] for i, pred in enumerate(predictions)]
    data = np.concatenate([[["Bound"]], data])

    data_frame = pd.DataFrame(data=data[1:, :], columns=data[0])
    data_frame.index.name = "Id"
    data_frame.to_csv(out_fname)


def kernel_train(kernel, training_data, ytrain, lmda):

    K = kernel.gram(training_data)
    alpha = LargeMargin.SVM(K, ytrain, lmda)
    return alpha


def kernel_predict(kernel, alpha, training, test):

    predict = []
    for x in tqdm(test):
        predict.append(np.sign(kernel.eval_f(x, alpha, training)))
    return predict


def score(predict, yreal):

    return sum([int(predict[i] == yreal[i]) for i in range(len(yreal))]) / len(yreal)


def split_data(dataset, y, k, m):

    dataset.compute_k_mers(k)
    dataset.mismatch(k, m)
    idx = range(len(dataset.data))
    pairs = []
    data_tranches = [idx[500 * i : 500 * i + 500] for i in range(4)]
    label_tranches = [y[500 * i : 500 * i + 500] for i in range(4)]
    for i in range(4):
        test, ytest = data_tranches[i], label_tranches[i]
        train = np.concatenate([data_tranches[j] for j in range(4) if j != i])
        ytrain = np.concatenate([label_tranches[j] for j in range(4) if j != i])

        pairs.append((train, ytrain, test, ytest))
    return pairs
