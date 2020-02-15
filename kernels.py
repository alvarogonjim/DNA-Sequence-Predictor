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


def scalar_product(x, y):
    """
    Scalar product
    """
    prod_scal = 0
    for idx in x:
        if idx in y:
            prod_scal += x[idx] * y[idx]
            # print(x[idx])
    return prod_scal

def sparse_gaussian(x, y, sigma):
    """
    Sparse gaussian
    """
    norm = scalar_product(x, x) - 2 * scalar_product(x, y) + scalar_product(y, y)
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-norm / (2 * sigma ** 2))

def gram_matrix(data, kernel, normalized=False):
    """
    Compute Gram matrix
    """
    n = len(data)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            prod_scal = kernel(data[i], data[j])
            K[i, j] = prod_scal
            K[j, i] = prod_scal

    if normalized:
        diag = np.sqrt(np.diag(K))
        for i in range(n):
            K[i, :] = K[i, :] / diag[i]
            K[:, i] = K[:, i] / diag[i]

    return K

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
