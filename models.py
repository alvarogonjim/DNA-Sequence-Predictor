"""
@authors: GONZALEZ Alvaro: alvarogonjim95@gmail.com
        & LAURENDEAU Matthieu: laurendeau.matthieu@gmail.com
@date: 2020 January
@brief: Models defined and test for the kaggle challenge
"""

############ Imports ############
"""
Libraries necessary to run this file alone.
"""
import pandas as pd # for data management
import numpy as np # for operation on array
import matplotlib.pyplot as plt # for plots (overfitting)
from kernels import linear_kernel, rbf_kernel, polynomial_kernel # to use kernels
import time # to see computation time

############ Logistic Regression Model ############
class logistic_regression_model:
    """
    Class for the logisitc regression model.
    """
    def __init__(self, epochs):
        """
        Constructor of the model
        @param: epochs: int - number of epochs for the training
        """
        self.name = "logistic_regression_model"
        self.epochs = epochs
    
    def train(self, train_set, validation_set, label, plots=True):
        """
        Method to train the model.
        @param: train_set: pandas dataframe - training data set
                validation_set: pandas dataframe - validation data set
                label: pandas dataframe - label set
                plots: bool - whether plots the overfitting graph (1) or not (0). 
                Default = True.
        """
        # Save time to see run time computation
        time_start = time.clock()

        # Define weight for the model
        self.w = np.zeros(train_set.shape[1])

        # Get label for training and validation set
        label_training = label[label.index.isin(train_set.index)]
        label_validation = label[label.index.isin(validation_set.index)]

        # Process sets for numpy operations
        train_set = np.array(train_set)
        validation_set = np.array(validation_set)
        label_training = np.array(label_training.iloc[:,0])
        label_validation = np.array(label_validation.iloc[:,0])

        # For checking overfitting
        err_training = []
        err_validation = []

        # Turn on intercative graphic mode for online overfitting check
        if plots:
            fig = plt.figure()
            fig.canvas.set_window_title(self.name)
            plt.ion()
            plt.show()

        for t in range(self.epochs):
            print("Epochs: " + str(t) + "/" + str(self.epochs), end="\r")

            # TO DO
        
            # Prediciton over the training set
            pred_training = self.predict(train_set)
            # Evaluation of the prediciton over the training set
            err_training += [self.evaluate(pred_training,label_training)]
            # Prediciton over the validation set
            pred_validation = self.predict(validation_set)
            # Evaluation of the prediciton over the validation set
            err_validation += [self.evaluate(pred_validation,label_validation)]
            # Treshold overfitting

            if plots:
                # Plot overfitting graph online
                plt.clf()
                plt.plot(range(t+1), err_training, label="Error training")
                plt.plot(range(t+1), err_validation, label="Error validation")
                plt.xlabel("Epochs")
                plt.ylabel("Error")
                plt.title("Overfitting plot: training vs validation error " \
                            + "prediction")
                plt.legend()
                plt.pause(0.5)
        
        print("Save overfitting graph.")
        plt.savefig("overfitting_"+self.name+".png")

        # Calculate run time computation
        time_elapsed = (time.clock() - time_start)
        print("Run time training: " + str(time_elapsed) + "s.")

    def predict(self, data):
        """
        Predict the model over data.
        @param: data: numpy array - data to predict
        """

        # TO DO
        predictions = np.dot(self.w, data.T)

        return 1*(predictions>0)

    def evaluate(self, prediction, label, save=True):
        """
        Evaluate the error of prediction versus label.
        @param: prediction: numpy array - data predicted
                label: numpy array - label set corresponding to the 
                data predicted.
        """
        error = 0

        for i in range(len(prediction)):
            error += abs(prediction[i] - label[i])

        return round(error*100/len(prediction), 2)

    def save(self):
        """
        Save the model.
        """
        print("Save model.")
        weights = pd.DataFrame({"Weights": self.w})
        weights.index.name = 'Id'
        weights.to_csv(self.name+".csv", sep=",", encoding="utf-8", index=True)



############ Support vector machine model ############

import cvxopt
import cvxopt.solvers
from collections import Counter
from itertools import combinations_with_replacement
class SVM(object):
    def __init__(self, kernel=polynomial_kernel, C=1):
        self.kernel = kernel
        self.C = C

    def train(self, train_set, validation_set, label, plots=False):
        # USE VALIDATION SET FOR CHECK OVERFITTING

        # Get label for training and validation set
        y = label[label.index.isin(train_set.index)]
        label_validation = label[label.index.isin(validation_set.index)]

        # Process sets for numpy operations
        x = np.array(train_set)
        validation_set = np.array(validation_set)
        y = np.array(y.iloc[:,0])
        label_validation = np.array(label_validation.iloc[:,0])

        num_obs = x.shape[0]
        # Gram matrix
        Gram = np.zeros((num_obs, num_obs))
        print("Computing Gram matrix:")
        for i in range(num_obs):
            if (i % 100 == 0):
                print(i, "/", num_obs)
            for j in range(num_obs):
                Gram[i, j] = self.kernel(x[i], x[j])

        # Components for quadratic program problem
        P = cvxopt.matrix(np.outer(y, y) * Gram)
        q = cvxopt.matrix(-np.ones((num_obs, 1)))
        A = cvxopt.matrix(y, (1, num_obs), 'd') # 6000 x 1024 , (1,6000)
        b = cvxopt.matrix(np.zeros(1)) # 0
        diag = np.diag(np.ones(num_obs) * -1)
        identity = np.identity(num_obs)
        G = cvxopt.matrix(np.vstack((diag, identity)))
        h = cvxopt.matrix(np.hstack((np.zeros(num_obs), np.ones(num_obs) * self.C)))

        # Solving quadratic progam problem
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])

        # Support vectors have non zero lagrange multipliers, cut off at 1e-6
        sup_vec = alphas > 1e-6
        ind = np.arange(len(alphas))[sup_vec]

        # Creating support vectors
        self.alphas = alphas[sup_vec]
        self.sup_vec = x[sup_vec]
        self.sup_vec_y = y[sup_vec]

        # Fitting support vectors with the intercept
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.sup_vec_y[i]
            self.b -= np.sum(self.alphas * self.sup_vec_y * Gram[ind[i], sup_vec])
        self.b /= len(self.alphas)
        print(self.b)

        # Weight for non linear kernel(polynomial or rbf)
        self.w = None

        # Predict the sign

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alphas, sup_vec_y, sup_vec in zip(self.alphas, self.sup_vec_y, self.sup_vec):
                s += alphas * sup_vec_y * self.kernel(X[i], sup_vec)
            y_pred[i] = s

        predictions = y_pred + self.b
        return 1*(predictions>0)

    def save(self):
        """
        Save the model.
        """
        print("Save model.")
        # TO DO

############ Main ############
''' If the file is executed separetely '''
if __name__ == "__main__":
    print("This file need train data and test data set at least to be run.")