# -*- coding:utf-8 -*-
from pyspark.ml.classification import LogisticRegression
from AutoML.helper.check import check_truth_pred_shape, check_label, check_params

class LogisticRegressionClassifier(object):
    def __init__(self, args=None):
        self.lr = LogisticRegression()
        if check_params(args):
            self.args = dict(args)
            self.lr.setParams(**self.args)

    def fit(self, X, y, sample_weight=None):
        check_label(y)
        self.sample_size = X.shape
        self.lr.fit(X, y, sample_weight=sample_weight)


