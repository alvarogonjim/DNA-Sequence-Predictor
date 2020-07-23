__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_kernels__ = "Kernels definition."

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import numpy as np  # for arrays tricks


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


def scalar_product(x, y):
    """
    Scalar product
    """
    res = 0
    for idx in x:
        if idx in y:
            res += x[idx] * y[idx]
    return res


def gaussian(x, y, sigma=6):
    """
    Gaussian kernel
    """
    return (
        1
        / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
    )


def sparse_gaussian(x, y, sigma=7.8):
    norm = scalar_product(x, x) - 2 * scalar_product(x, y) + scalar_product(y, y)
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-norm / (2 * sigma ** 2))


def gram_matrix(train_set_kmer, validation_set_kmer, test_set_kmer, kernel):
    """
    Compute Gram matrix on all the datasets
    """
    data = train_set_kmer + validation_set_kmer + test_set_kmer
    n = len(data)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            print(
                str(round((i * n + j) / (n * n) * 100)).zfill(3) + "%", end="\b\b\b\b"
            )
            K = kernel(data[i], data[j])
            G[i, j] = K
            G[j, i] = K
    return G
