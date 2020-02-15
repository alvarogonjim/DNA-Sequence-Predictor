__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_utils__ = "This file contains some util methods that we used during the execution' \
        like write the predictions in a CSV or get the score of the model during the training"

import numpy as np
import pandas as pd

def write_predictions(predictions, out_fname='Yte.csv'):
    '''
    @param predictions: Numpy array which contains all the results that our model has predicted.
    @param out_fname: Name of the file to store the results by default is Yte.csv
    '''
    data = [[int(np.abs((pred + 1) // 2))] for i, pred in enumerate(predictions)]
    data = np.concatenate([[["Bound"]], data])

    data_frame = pd.DataFrame(data=data[1:, :], columns=data[0])
    data_frame.index.name = "Id"
    data_frame.to_csv(out_fname)

def score(predict, yreal):
    '''
    @param predict; Numpy array with all the results that our model has predicted.
    @param yreal: Numpy array with the real labels of the sequences.
    '''
    return sum([int(predict[i] == yreal[i]) for i in range(len(yreal))]) / len(yreal)
