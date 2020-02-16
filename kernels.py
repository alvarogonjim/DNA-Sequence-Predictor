__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_compute_kmer_feature__ = "Kernels definition."

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import numpy as np # for arrays tricks
from multiprocessing import Pool, Manager # for multiprocess the code
from functools import partial # for create partial objec

def linear_kernel(x, y):
    """
    Linear kernel
    """
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    """
    Polynomial kernel
    """
    return (1 + np.dot(x, y)) ** p

def rbf_kernel(x, y, sigma=3):
    """
    Radial basis function kernel
    """
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def gaussian(x, y, sigma):
    """
    Gaussian kernel
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma) \
    * np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2)))


def scalar_product_hpc(x, y):
    """
    Scalar product for multiprocessing
    """
    print(y)
    assert(False)
    x = d[0]
    y = d[1]
    if x[0] == y[0]:
        res = x[1] * y[1]



def scalar_product(x, y):
    """
    Scalar product
    """
    res = 0
    for idx in x:
        if idx in y:
            res += x[idx] * y[idx]
    return res

def sparse_gaussian(x, y, sigma):
    """
    Sparse gaussian
    """
    norm = scalar_product(x, x) - 2 * scalar_product(x, y) + scalar_product(y, y)
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-norm / (2 * sigma ** 2))

def gram_matrix(train_set_kmer, validation_set_kmer, test_set_kmer, kernel):
    """
    Compute Gram matrix on all the datasets
    """
    data = train_set_kmer + validation_set_kmer + test_set_kmer
    # print(data[-3:])
    n = len(data)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            K = kernel(data[i], data[j])
            G[i, j] = K
            G[j, i] = K
    return G

# def gram_matrix(train_set_kmer, validation_set_kmer, test_set_kmer, kernel):
#     """
#     Compute Gram matrix on all the datasets
#     """
#     data = train_set_kmer + validation_set_kmer + test_set_kmer
#     n = len(data)
#     G = np.zeros((n, n))
#     for i in range(n):
#         d = Manager().dict()
#         d = data[i+1]
#         # print(d)
#         # assert(False)
#         K = 0
#         pool = Pool(processes=8)
#         for result in pool.imap_unordered(partial(scalar_product_hpc, \
#             data[i]), d):
#             K += result
#         # K = kernel(data[i], data[j])
#         G[i, j] = K
#         G[j, i] = K
#     return G

def eval_f(self, x, alpha, data):
    if self.normalized:
        square_norm_x = np.sqrt(self.kernel(x, x))
        result = np.sum(
            [
                (alpha[i] * self.kernel(x, xi)) / (square_norm_x * self.diag[i])
                for i, xi in enumerate(data)
            ]
        )
    else:
        result = np.sum(
            [alpha[i] * self.kernel(x, xi) for i, xi in enumerate(data)]
        )
    return result
