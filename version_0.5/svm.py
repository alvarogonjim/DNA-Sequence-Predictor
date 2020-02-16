__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_svm__ = "Class that contains the implementation of the \
    Support Vector Machine"


############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import numpy as np # for arrays operations
from cvxopt import solvers, matrix, spmatrix, sparse # convex optimization
solvers.options["show_progress"] = False

class SVM:
    """
    Class Support Vector Machine
    """
    def __init__(self, l):
        """
        Construct SVM model
        @param: lmda: float - Hyperparameter lambda
        """
        self.l = l

    def fit(self, gram_matrix, labels):
        """
        Solve:  min_x 1/2 xPx +qx
            s.t Gx <= h
                Ax = b
        @param: gram_matrix: numpy array - Gram Matrix
                labels: numpy array - Contains all the labels (-1, 1) to fit 
        the SVM
        """

        # Components for quadratic program problem
        n = len(labels)
        P = matrix(gram_matrix)
        q = -matrix(labels, (n, 1), tc="d")

        # Constraints
        G = spmatrix(labels, range(n), range(n)) # diagonal matrix
        G = sparse([G, -G])
        h = np.concatenate([np.ones(n) / (2 * self.l * n), np.zeros(n)])
        h = matrix(h.reshape((2 * n, 1)))

        # Solving quadratic progam problem
        self.alpha = solvers.qp(P, q, G, h)["x"]

        # Return the solution
        return self.alpha

    def predict(self, K):
        """
        Predict class from SVM model
        @param: K: numpy array - kernel
        """
        predictions = []
        # assert False, np.shape(K)[1]
        for i in range(np.shape(K)[0]):
            # print("i", i)
            pred = 0
            for k, j in enumerate(range(np.shape(K)[1])):
                # print("k", k, "j", j)
                # print(K[i, j])
                pred += self.alpha[k] * K[i, j]
            # assert(False)
            predictions.append(np.sign(pred))
        return predictions

def score(predict, label):
    '''
    Evaluate performances of the model. Compare predictions and true label
    @param: predict: numpy array - predictions
            label: numpy array - real labels of the sequences
    '''
    res = 0
    for i in range(len(label)):
        res += int(predict[i] == label[i]) / len(label)
    return res
