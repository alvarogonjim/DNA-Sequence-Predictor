import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Kernel_KMeans(object):

    def __init__(self,k=2):
        self.k = k
        self.classes = {}
        self.factor = {}

    def kernel_distance(self,x, classification):
        l = len(self.classes[classification])
        res = self.kernel(x,x)- \
              2*np.sum([self.kernel(x,j) for j in self.classes[classification]])/l+self.factor[classification]


        return res

    def _gaussian(self,sigma):
        gamma = 1/1024
        def f(x,y):
            return np.exp(-1/1024*np.linalg.norm(x-y)**1)
            #return (gamma*np.inner(x,y))**2
            #return sigmoid(gamma*np.inner(x,y))
            #return np.inner(x,y)
        return f

    def fit(self,X,y):
        self.kernel = self._gaussian(1)
        self.centroids = np.zeros((self.k,X.shape[1]))
        # initialize the centroids
        for i in range(self.k):
            self.centroids[i] = X[i]

        for i in range(self.k):
            self.classes[i] =  []
        # find the distance bw the point and cluster
        # choose the nearest centroid
        for i,features in enumerate(X):
            self.classes[y[i]].append(features)
        for classification in self.classes:
            self.centroids[classification] = np.average(self.classes[classification], axis =0)

        for ii in range(self.k):
            l = len(self.classes[ii])
            self.factor[ii] = np.sum([np.sum([self.kernel(i,j) for j in self.classes[ii]]) for i in self.classes[ii]])/(l**2)


    def predict(self,X):
        res = np.zeros((X.shape[0]))
        for i,features in enumerate(X):
                distances = [self.kernel_distance(features,classification) for classification in range(0,len(self.classes))]
                classification = distances.index(min(distances))
                res[i] = classification
        return res

    def score(self,X,y):
        preds = self.predict(X)
        return (preds == y).sum().astype(float)/len(preds)
