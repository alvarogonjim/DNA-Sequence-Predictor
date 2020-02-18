__author__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__date__ = "2020 January"
__copyright__ = "Copyright 2020, Advanced Learning Models"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Gonzalez Jimenez Alvaro, Laurendeau Matthieu"
__email__ = "alvaro.gonzalez-jimenez@grenoble-inp.org, laurendeau.matthieu@gmail.com"
__status__ = "Submitted"
__brief_start__ = "Required start file to reproduce the Kaggle submission"

from main import predict_dataset, save_result

########### Predictions ############
dataset_numbers = [0, 1, 22]
kmer_length = [[9, 10, 11]]
mismatch = [[1, 1, 1]]
lambdas = [0.000001]
predictions = predict_dataset(dataset_numbers, kmer_length, mismatch, lambdas)

############ Save results ############
submit = predictions[1] + predictions[2] + predictions[3]
save_result(submit, "Yte_GONZALEZ_LAURENDEAU_kaggle_submission.csv")