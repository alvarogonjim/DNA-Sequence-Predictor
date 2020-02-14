#!/usr/bin/env python

__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org"
__status__ = "Develop"

import numpy as np


class logistic_regression(object):
    def __init__(self, num_steps=1000, lr=0.01):

        self.num_steps = num_steps
        self.lr = lr

    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))

    def log_likelihood(self, X, y, weight):
        scores = np.dot(X, self.weights)
        l1 = np.sum(y * scores - np.log(1 + np.exp(scores)))
        return l1

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for i in range(0, self.num_steps):
            scores = np.dot(X, self.weights)
            predictions = self.sigmoid(scores)

            output_error = y - predictions
            gradient = np.dot(X.T, output_error)
            self.weights += self.lr / (10 ** (int)(i / 2000)) * gradient

            # if i % 1000 == 0:
            # print("Iteration %d : log_likelihood = %f"%(i,self.log_likelihood(X,y,self.weights)))

    def score(self, X, y):
        preds = self.predict(X)
        return (preds == y).sum().astype(float) / len(preds)

    def predict(self, X):
        scores = np.dot(X, self.weights)
        raw_predicts = self.sigmoid(scores)
        predicts = np.zeros(X.shape[0])
        predicts[np.where(raw_predicts > 0.5)] = 1
        return predicts
