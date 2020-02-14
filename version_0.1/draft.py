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

class logistic_regression_model_v1:
    def __init__(self):
        self.name = "logistic_regression_model_v1"
        self.n = 0.003
        self.epochs = 100
        self.lamb = 0.1
    
    def train(self, train_set, validation_set, label, plots=True):
        self.w0 = 0
        self.w = np.zeros(train_set.shape[1])
        t = 0
        
        # Get label for training and validation set
        label_training = label[label.index.isin(train_set.index)]
        label_validation = label[label.index.isin(validation_set.index)]

        # Process set for numpy operations
        train_set = np.array(train_set)
        validation_set = np.array(validation_set)
        label_training = np.array(label_training.iloc[:,0])
        label_validation = np.array(label_validation.iloc[:,0])

        # Check overfitting
        err_training = []
        err_validation = []
        while t<self.epochs:
            print("Epochs: " + str(t) + "/" + str(self.epochs), end="\r")
            hw = self.w0 + np.dot(train_set,self.w)
            hw = 1/(1+np.exp(-hw))
            grad = np.dot(label_training-hw,train_set) - self.lamb*self.w
            self.w = self.w + self.n*grad
            self.w0 = self.w0 + self.n*np.sum(label_training-hw) - self.n*self.lamb
            if np.linalg.norm(grad)<0.1: break
            t+=1
        

            err_training += [self.evaluate(self.predict(train_set),label_training)]
            err_validation += [self.evaluate(self.predict(validation_set),label_validation)]

            # No condition overfitting

        if plots:
            plt.plot(range(t), err_training, label="Error training")
            plt.plot(range(t), err_validation, label="Error validation")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.title("Overfitting plot: training vs validation error prediction")
            plt.legend()
            plt.savefig("overfitting_"+self.name+".png")

    def predict(self, data):
        hw = self.w0 + self.w.dot(data.T)
        hw = 1/(1+np.exp(-hw))
        preds = (hw > 0.5)*1
        return preds

    def evaluate(self, prediction, label, save=True):
        error = 0

        for i in range(len(prediction)):
            error += abs(prediction[i] - label[i])

        return round(error*100/len(prediction), 2)

    def save(self):
        weights = pd.DataFrame({"Weights": np.concatenate([[self.w0],self.w])})
        weights.index.name = 'Id'
        weights.to_csv(self.name+".csv", sep=",", encoding="utf-8", index=True)


# def logistic_regression_model(train_set, label, epochs=5000, \
#                         n=0.003, lamb = 0.1):