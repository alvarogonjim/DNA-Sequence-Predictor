#!/usr/bin/env python

__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org"
__status__ = "Develop"


import pandas as pd
import numpy as np
from cvxopt import solvers, matrix, spmatrix, sparse
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
from string_kernel import ssk


def SVM(K, y, lmda):
    print("Optimizing")
    solvers.options["show_progress"] = False

    n = len(y)
    q = -matrix(y, (n, 1), tc="d")
    h = matrix(
        np.concatenate([np.ones(n) / (2 * lmda * n), np.zeros(n)]).reshape((2 * n, 1))
    )
    P = matrix(K)
    Gtop = spmatrix(y, range(n), range(n))
    G = sparse([Gtop, -Gtop])

    sol = solvers.qp(P, q, G, h)["x"]
    return sol


def substring_kernel(X, k, lam, norm=False):
    n = X.shape[0]
    gram = np.zeros((n, n))
    if norm:
        kxx_val = {}
        for i in range(n):
            kxx_val[i] = ssk(X[i], X[i], k, lam)
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = ssk(X[i], X[j], k, lam)
            if norm:
                gram[i, j] = gram[i, j] / (kxx_val[i] * kxx_val[j]) ** 0.5
            gram[j, i] = gram[i, j]
    return gram


def f_ssk(x, X, alpha, k, lam):
    out = sum([alpha[i] * ssk(x, X[i], k, lam) for i in range(X.shape[0])])
    return out


def evaluateSVM(K, lamSSK, lamSVM, X_train, Y_train, X_val, Y_val):
    alpha = SVM(K, Y_train, lamSVM)
    s = 0
    s_plus = 0
    s_moins = 0
    for i in range(Y_val.shape[0]):
        f_val = f_ssk(X_val[i], X_train, alpha, k, lamSSK)
        if f_val > 0 and Y_val[i] == 1:
            s_plus += 1
            s += 1
        if f_val < 0 and Y_val[i] == -1:
            s_moins += 1
            s += 1
    return s, s_plus, s_moins


train_string = pd.read_csv("./Xtr2.csv", sep=",", header=None)
X = train_string.values[1:, 1]
label = pd.read_csv("./Ytr2.csv")
Y = label["Bound"].values
for i in range(2000):
    if Y[i] == 0:
        Y[i] = -1

test_string = pd.read_csv("./Xte2.csv", sep=",", header=None)
Xtest = test_string.values[1:, 1]

X = np.concatenate((X, Xtest))

k = 9  # 8 8 9
lamSSK = 0.01  # 0.01 0.1 0.01
K = substring_kernel(X, k, lamSSK, norm=False)
np.save("K2.npy", K)
