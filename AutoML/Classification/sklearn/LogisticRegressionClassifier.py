# -*- coding:utf-8 -*-
"""This is used for building sklearn classfication classifier.
   The classifier chosen based on auto-sklearn paper:
    http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf
"""
from sklearn import linear_model
from AutoML.helper.check import check_params, check_label, check_truth_pred_shape

class LogisticRegression(object):
    def __init__(self, args=None):
        self.lr = linear_model.LogisticRegression()
        if check_params(args):
            self.args = dict(args)
            self.lr.set_params(**self.args)    # Set algorithm's parameters

    def fit(self, X, y, sample_weight=None):
        check_label(y)
        self.sample_size = X.shape
        self.lr.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.lr.predict(X)

    def predict_proba(self, X):
        check_truth_pred_shape(self.sample_size, X.shape)
        return self.lr.predict_proba(X)

    def score(self, X, y):
        check_label(y)
        return self.lr.score(X, y)



if __name__ == '__main__':
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)

    lr = LogisticRegression(None)
    lr.fit(x, y)
    print('Model score:', lr.score(x,  y))