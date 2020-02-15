__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_large_margin__ = "Class that contains the implementation of the Support Vector Machine"

from cvxopt import solvers, matrix, spmatrix, sparse
import numpy as np


class SVM:
    def fit(gram_matrix, labels, lmda):
        '''
        @param gram_matrix: Gran Matrix
        @param labels: Contains all the labels (-1, 1) to fit the SVM
        @param lmda: Hyperparameter lambda
        '''

        #Option to not show the progress of the Cvxopt
        solvers.options["show_progress"] = False

        # Components for quadratic program problem
        n = len(labels)
        q = -matrix(labels, (n, 1), tc="d")
        h = matrix(
            np.concatenate([np.ones(n) / (2 * lmda * n), np.zeros(n)]).reshape(
                (2 * n, 1)
            )
        )
        P = matrix(gram_matrix)
        Gtop = spmatrix(labels, range(n), range(n))
        G = sparse([Gtop, -Gtop])

        #Solving quadratic progam problem
        sol = solvers.qp(P, q, G, h)["x"]

        # Return the solution
        return sol
