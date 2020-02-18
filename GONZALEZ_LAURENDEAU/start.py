__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_start__ = "Required start file to reproduce the Kaggle submission"

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
from main import predict_dataset, save_result
from kernels import scalar_product, gaussian

########### Predictions ############
dataset_numbers = [0, 1, 2]
kmer_length = [[9, 10, 11], [9, 10, 11], [6]]
mismatch = [[1, 1, 1], [1, 1, 1], [1]]
lambdas = [0.001, 0.001, 0.00000001]
kernels = [scalar_product, scalar_product, gaussian]
predictions = predict_dataset(dataset_numbers, kmer_length, mismatch, lambdas, \
kernels, truncation = -1, shuffle = False, validation = 0)

############ Save results ############
submit = predictions[0] + predictions[1] + predictions[2]
save_result(submit, "Yte.csv")