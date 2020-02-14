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
from SVM import SVM
from tqdm import tqdm


def write_predictions(predictions, filename='Yte.csv'):
    '''
    @param: predictions: The predictions given by the model
    @param filename: Name of the file to store the predictions by default Yte.csv
    Method to write the prediction to submit it to Kaggle.
    '''
    data = [[int(np.abs((pred + 1) // 2))] for i, pred in enumerate(predictions)]
    data = np.concatenate([[["Bound"]], data])
    data_frame = pd.DataFrame(data=data[1:, :], columns=data[0])
    data_frame.index.name = "Id"
    data_frame.to_csv(filename)


def train(kernel, training_data, labels, lambda_param):
    '''
    @param kernel: Kernel method that we want to use
    @param training_data: K-mers for the training
    @param labels: The Y labels for the training data (Bound)
    @param lambda_param: The hyperparamater lambda

    Method to train a SVM given a kernel and the training data
    '''
    gram_matrix = kernel.gram(training_data)
    model = SVM.fit(gram_matrix, labels, lambda_param)
    return model


def predict(kernel, model, training, test):
    '''
    @param kernel: Kernel method that we want to use
    @param training_data: K-mers for the training
    @param labels: The Y labels for the training data (Bound)
    @param lambda_param: The hyperparamater lambda

    Method to train a SVM given a kernel and the training data
    '''
    predictions = []
    for x in tqdm(test):
        predictions.append(np.sign(kernel.eval_f(x, model, training)))
    return predictions


def score(prediction, real_label):
    '''
    @param prediction: List of predictions given by the model
    @param real_label: List of real labels

    Method to evaluate the model given the prediction and the real label
    '''

    return sum([int(prediction[i] == real_label[i]) for i in range(len(real_label))]) / len(real_label)


def split_data(dataset, y, k, m):
    '''
    @param dataset: Collection of data that we are going to split
    @param y: Collection of labels corresponding to the dataset
    @param k: Size of the k-mers
    @param m: Threshold of the mismatch

    Method to split the training data in train and validation set.
    Compute the k-mers and the mismatch method, then divide the data.
    '''
    dataset.compute_k_mers(k)
    dataset.mismatch(k, m)
    index = range(len(dataset.data))
    result = []
    data_sections = [index[500 * i : 500 * i + 500] for i in range(4)]
    label_sections = [y[500 * i : 500 * i + 500] for i in range(4)]
    for i in range(4):
        test, ytest = data_sections[i], label_sections[i]
        train = np.concatenate([data_sections[j] for j in range(4) if j != i])
        ytrain = np.concatenate([label_sections[j] for j in range(4) if j != i])
        result.append((train, ytrain, test, ytest))
    return result
