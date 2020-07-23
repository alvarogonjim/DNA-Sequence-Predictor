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
import pandas as pd  # for data management
import numpy as np  # for operation on array
import matplotlib.pyplot as plt  # for plots (overfitting)
from kernels import linear_kernel, rbf_kernel, polynomial_kernel  # to use kernels
import time  # to see computation time

############ Sigmoid function ############
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_pred, target):
    return -np.mean((target * np.log(y_pred) + (1 - target) * np.log(1 - y_pred)))


############ Logistic Regression Model ############
class logistic_regression_model:
    """
    Class for the logisitc regression model.
    """

    def __init__(self, epochs, learning_rate=0.1):
        """
        Constructor of the model
        @param: epochs: int - number of epochs for the training
        """
        self.name = "logistic_regression_model"
        self.epochs = epochs
        self.learning_rate = learning_rate

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
        self.w = np.random.uniform(0, 1, size=(train_set.shape[1]))  # weights
        self.b = 0.5  # bias

        # Get label for training and validation set
        label_training = label[label.index.isin(train_set.index)]
        label_validation = label[label.index.isin(validation_set.index)]

        # Process sets for numpy operations
        train_set = np.array(train_set)
        validation_set = np.array(validation_set)
        label_training = np.array(label_training.iloc[:, 0])
        label_validation = np.array(label_validation.iloc[:, 0])

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
            print("Epochs: " + str(t) + "/" + str(self.epochs), end="")

            Z = np.dot(train_set, self.w) + self.b
            Y_output = sigmoid(Z)[0]
            E = cross_entropy_loss(Y_output, label_training)
            grad = Y_output - label_training
            grad_weight = np.dot(train_set.T, grad) / train_set.shape[0]
            grad_bias = np.average(grad)
            self.w = self.w - self.learning_rate * grad_weight
            self.b = self.b - self.learning_rate * grad_bias

            print(" | loss = " + str(round(E, 2)))

            # Prediciton over the training set
            pred_training = self.predict(train_set)
            # Evaluation of the prediciton over the training set
            err_training += [self.evaluate(pred_training, label_training)]
            # Prediciton over the validation set
            pred_validation = self.predict(validation_set)
            # Evaluation of the prediciton over the validation set
            err_validation += [self.evaluate(pred_validation, label_validation)]
            # Treshold overfitting

            if plots:
                # Plot overfitting graph online
                plt.clf()
                plt.plot(range(t + 1), err_training, label="Error training")
                plt.plot(range(t + 1), err_validation, label="Error validation")
                plt.xlabel("Epochs")
                plt.ylabel("Error")
                plt.title(
                    "Overfitting plot: training vs validation error " + "prediction"
                )
                plt.legend()
                plt.pause(0.5)

        print("Save overfitting graph.")
        plt.savefig("overfitting_" + self.name + ".png")

        # Calculate run time computation
        time_elapsed = time.clock() - time_start
        print("Run time training: " + str(time_elapsed) + "s.")

    def predict(self, data):
        """
        Predict the model over data.
        @param: data: numpy array - data to predict
        """

        predictions = []
        for i in sigmoid(np.dot(data, self.w) + self.b):
            if i > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def evaluate(self, prediction, label, save=True):
        """
        Evaluate the error of prediction versus label.
        @param: prediction: numpy array - data predicted
                label: numpy array - label set corresponding to the 
                data predicted.
        """
        # TO DO: F1 score
        error = 0

        for i in range(len(prediction)):
            error += abs(prediction[i] - label[i])

        return round(error * 100 / len(prediction), 2)

    def save(self):
        """
        Save the model.
        """
        print("Save model.")
        weights = pd.DataFrame({"Weights": self.w})
        weights.index.name = "Id"
        weights.to_csv(self.name + ".csv", sep=",", encoding="utf-8", index=True)


############ Gaussian process classifier ############
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class gaussian_process_clissifier:
    """
    Class for the logisitc regression model.
    """

    def __init__(self):
        """
        Constructor of the model
        @param: epochs: int - number of epochs for the training
        """
        self.model = GaussianProcessClassifier(1.0 * RBF(1.0))
        self.name = "gaussian_process_clissifier"
        # self.model = SVC(kernel = 'linear')
        # self.name = "SVM_linear"

    def train(self, train_set, validation_set, label, plots=True):
        """
        Method to train the model.
        @param: train_set: pandas dataframe - training data set
                validation_set: pandas dataframe - validation data set
                label: pandas dataframe - label set
                plots: bool - whether plots the overfitting graph (1) or not (0). 
                Default = True.
        """
        # Get label for training and validation set
        label_training = label[label.index.isin(train_set.index)]
        label_validation = label[label.index.isin(validation_set.index)]

        # Process sets for numpy operations
        train_set = np.array(train_set)
        validation_set = np.array(validation_set)
        label_training = np.array(label_training.iloc[:, 0])
        label_validation = np.array(label_validation.iloc[:, 0])

        self.model.fit(train_set, label_training)
        predictions_validation = self.model.predict(validation_set)
        print(
            "Test-- ",
            self.name,
            ": ",
            accuracy_score(label_validation, predictions_validation),
        )

    def predict(self, data):
        """
        Predict the model over data.
        @param: data: numpy array - data to predict
        """
        return self.model.predict(data)

    def evaluate(self, prediction, label, save=True):
        """
        Evaluate the error of prediction versus label.
        @param: prediction: numpy array - data predicted
                label: numpy array - label set corresponding to the 
                data predicted.
        """
        # TO DO: F1 score
        error = 0

        for i in range(len(prediction)):
            error += abs(prediction[i] - label[i])

        return round(error * 100 / len(prediction), 2)

    def save(self):
        """
        Save the model.
        """
        print("Save model.")


############ Support vector machine model ############
# NEED TO BE CHANGED ACCORDING THE NEW FEATURES K-MER DEFINITION
# NOW IT IS A BIG BINARY MATRIX.
import cvxopt
import cvxopt.solvers
from collections import Counter
from itertools import combinations_with_replacement


class SVM(object):
    def __init__(self, kernel=polynomial_kernel, C=1):
        self.kernel = kernel
        self.C = C
        self.name = "SVM"

    def train(self, train_set, validation_set, label, plots=False):
        # Save time to see run time computation
        time_start = time.clock()

        # Get label for training and validation set
        label_training = label[label.index.isin(train_set.index)]
        label_validation = label[label.index.isin(validation_set.index)]

        # Process sets for numpy operations
        train_set = np.array(train_set)
        validation_set = np.array(validation_set)
        label_training = np.array(label_training.iloc[:, 0])
        label_validation = np.array(label_validation.iloc[:, 0])

        num_obs = train_set.shape[0]
        # Gram matrix
        Gram = np.zeros((num_obs, num_obs))
        print("Computing Gram matrix:")
        for i in range(num_obs):
            if i % 100 == 0:
                print(i, "/", num_obs)
            for j in range(num_obs):
                Gram[i, j] = self.kernel(train_set[i], train_set[j])

        # Components for quadratic program problem
        P = cvxopt.matrix(np.outer(label_training, label_training) * Gram)
        q = cvxopt.matrix(-np.ones((num_obs, 1)))
        A = cvxopt.matrix(label_training, (1, num_obs), "d")  # 6000 x 1024 , (1,6000)
        b = cvxopt.matrix(np.zeros(1))  # 0
        diag = np.diag(np.ones(num_obs) * -1)
        identity = np.identity(num_obs)
        G = cvxopt.matrix(np.vstack((diag, identity)))
        h = cvxopt.matrix(np.hstack((np.zeros(num_obs), np.ones(num_obs) * self.C)))

        # Solving quadratic progam problem
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol["x"])

        # Support vectors have non zero lagrange multipliers, cut off at 1e-6
        sup_vec = alphas > 1e-6
        ind = np.arange(len(alphas))[sup_vec]

        # Creating support vectors
        self.alphas = alphas[sup_vec]
        self.sup_vec = train_set[sup_vec]
        self.sup_vec_y = label_training[sup_vec]

        # Fitting support vectors with the intercept
        self.b = 0
        for i in range(len(self.alphas)):
            self.b += self.sup_vec_y[i]
            self.b -= np.sum(self.alphas * self.sup_vec_y * Gram[ind[i], sup_vec])
        self.b /= len(self.alphas)
        print(self.b)

        # Weight for non linear kernel(polynomial or rbf)
        self.w = None

        pred_validation = self.predict(validation_set)
        print(
            "Error training: " + str(self.evaluate(pred_validation, label_validation))
        )

        # Calculate run time computation
        time_elapsed = time.clock() - time_start
        print("Run time training: " + str(time_elapsed) + "s.")

    # Predict the sign
    def predict(self, data):
        y_pred = np.zeros(len(data))
        for i in range(len(data)):
            s = 0
            for alphas, sup_vec_y, sup_vec in zip(
                self.alphas, self.sup_vec_y, self.sup_vec
            ):
                s += alphas * sup_vec_y * self.kernel(data[i], sup_vec)
            y_pred[i] = s

        predictions = y_pred + self.b
        return 1 * (predictions > 0)

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

        return round(error * 100 / len(prediction), 2)

    def save(self):
        """
        Save the model.
        """
        print("Save model.")
        # TO DO
        # weights = pd.DataFrame({"Weights": self.w, "Bias": self.b, \
        #             "Alphas": self.alphas, "Sup_vec": self.sup_vec, \
        #             "Sup_vec_y": self.sup_vec_y})
        # weights.index.name = 'Id'
        # weights.to_csv(self.name+".csv", sep=",", encoding="utf-8", index=True)


############ Main ############
""" If the file is executed separetely """
if __name__ == "__main__":
    print("This file need train data and test data set at least to be run.")
