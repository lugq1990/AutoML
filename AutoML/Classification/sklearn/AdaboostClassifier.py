# -*- coding:utf-8 -*-
from sklearn import ensemble
from AutoML.helper.check import *

class AdaboostClassifier(object):
    def __init__(self, args=None):
        self.abc = ensemble.AdaBoostClassifier()
        if check_params(args):
            self.args = dict(args)
            self.abc.set_params(**self.args)

    def fit(self, X, y, sample_weight=None):
        self.sample_size = X.shape
        check_label(y)
        self.abc.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.abc.predict(X)

    def predict_proba(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.abc.predict_proba(X)

    def score(self, X, y):
        check_label(y)
        return self.abc.score(X, y)