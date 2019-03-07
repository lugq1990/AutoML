# -*- coding:utf-8 -*-
from sklearn import ensemble
from AutoML.helper.check import *

class RandomForestClassifier(object):
    def __init__(self, args=None):
        self.rfc = ensemble.RandomForestClassifier()
        if check_params(args):
            self.args = dict(args)
            self.rfc.set_params(**self.args)

    def fit(self, X, y, sample_weight=None):
        check_label(y)
        self.sample_size = X.shape
        self.rfc.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.rfc.predict(X)

    def predict_proba(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.rfc.predict_proba(X)

    def score(self, X, y):
        return self.rfc.score(X, y)
