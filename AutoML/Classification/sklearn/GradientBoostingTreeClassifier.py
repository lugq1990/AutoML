# -*- coding:utf-8 -*-
from sklearn import ensemble
from AutoML.helper.check import *

class GradientBoostingTreeClassifier(object):
    def __init__(self, args):
        self.gbc = ensemble.GradientBoostingClassifier()
        if check_params(args):
            self.args = dict(args)
            self.gbc.set_params(**self.args)

    def fit(self, X, y, sample_weight=None):
        check_label(y)
        self.sample_size = X.shape
        self.gbc.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.gbc.predict(X)

    def predict_proba(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.gbc.predict_proba(X)

    def score(self, X, y):
        check_label(y)
        return self.gbc.score(X, y)