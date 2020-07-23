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


class KMeans(object):
    def __init__(self, k=2):
        self.k = k
        self.classes = {}

    def fit(self, X, y):
        self.centroids = np.zeros((self.k, X.shape[1]))
        # initialize the centroids
        for i in range(self.k):
            self.centroids[i] = X[i]

        for i in range(self.k):
            self.classes[i] = []
        # find the distance bw the point and cluster
        # choose the nearest centroid
        for i, features in enumerate(X):
            self.classes[y[i]].append(features)
        for classification in self.classes:
            self.centroids[classification] = np.average(
                self.classes[classification], axis=0
            )

    def predict(self, X):
        res = np.zeros((X.shape[0]))
        for i, features in enumerate(X):
            distances = [np.linalg.norm(features - x) for x in self.centroids]
            classification = distances.index(min(distances))
            res[i] = classification
        return res

    def score(self, X, y):
        preds = self.predict(X)
        return (preds == y).sum().astype(float) / len(preds)
