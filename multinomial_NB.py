import numpy as np

class multinomial_NB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self,X,y):
        separated = [[x for x,t in zip(X,y) if t==c] for c in np.unique(y)]
        count_sample = X.shape[0]
        self.class_log_prior = self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
            for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self,X,y):
        preds = self.predict(X)
        return (preds == y).sum().astype(float)/len(preds)
