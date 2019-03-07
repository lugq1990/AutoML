# -*- coding:utf-8 -*-
from sklearn import ensemble
from AutoML.helper.check import *

class ExtremeTreeClassifier(object):
    def __init__(self, args):
        self.etc = ensemble.ExtraTreesClassifier()
        if check_params(args):
            self.args = dict(args)
            self.etc.set_params(**self.args)

    def fit(self, X, y, sample_weight=None):
        check_label(y)
        self.sample_shape = X.shape
        self.etc.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_truth_pred_shape(self.sample_shape, X.shape)
        return self.etc.predict(X)

    def predict_proba(self, X):
        check_truth_pred_shape(self.sample_shape, X.shape)
        return self.etc.predict_proba(X)

    def score(self, X, y):
        check_label(y)
        return self.etc.score(X, y)