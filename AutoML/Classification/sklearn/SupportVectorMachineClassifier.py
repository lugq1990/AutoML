# -*- coding:utf-8 -*-
from sklearn.svm import SVC
from AutoML.helper.check import *

class SupportVectorClassifer(object):
    def __init__(self, args):
        self.svc = SVC()
        if check_params(args):
            self.args = dict(args)
            self.svc.set_params(**self.args)
        # Because SVM not supports predicted prob, so here must use decision boundary to predict prob
        self.svc.probability = True

    def fit(self, X, y, sample_weight=None):
        check_label(y)
        self.sample_size = X.shape
        self.svc.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.svc.predict(X)

    def predict_proba(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.svc.predict_proba(X)

    def score(self, X, y):
        check_label(y)
        return self.svc.score(X, y)