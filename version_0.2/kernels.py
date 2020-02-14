"""
@authors: GONZALEZ Alvaro: alvarogonjim95@gmail.com
        & LAURENDEAU Matthieu: laurendeau.matthieu@gmail.com
@date: 2020 January
@brief: Kernel definitions
"""

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import numpy as np # for data management


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

def linear_kernel(x, y):
    """
    Linear kernel
    """
    return np.inner(x, y)

############ Main ############
''' If the file is executed separetely '''
if __name__ == "__main__":
    print("This file simply define kernels.")