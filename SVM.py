#!/usr/bin/env python

__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org"
__status__ = "Develop"

from cvxopt import solvers, matrix, spmatrix, sparse
import numpy as np

class SVM:
    def fit(K, y, lambda_param):
        '''
        @param K: Kernel
        @param y: Label of the data (Bound)
        @param lambda_param: Hyperparameter lambda

        Compute solvers given the Kernel the label and the lambda
        '''
        solvers.options["show_progress"] = False
        n = len(y)
        q = -matrix(y, (n, 1), tc="d")
        h = matrix(
            np.concatenate([np.ones(n) / (2 * lambda_param * n), np.zeros(n)]).reshape(
                (2 * n, 1)
            )
        )
        P = matrix(K)
        Gtop = spmatrix(y, range(n), range(n))
        G = sparse([Gtop, -Gtop])
        sol = solvers.qp(P, q, G, h)["x"]
        return sol
