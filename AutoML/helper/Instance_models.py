# -*- coding:utf-8 -*-
""" This class is based on helper.GridParams returned result to instant the algorithm model or just instant
algorithm model based on given or default parameters. This class is mean instant object class!
"""
from sklearn.model_selection import GridSearchCV
from AutoML.Classification.sklearn import LogisticRegressionClassifier, AdaboostClassifier, \
    GradientBoostingTreeClassifier, SupportVectorMachineClassifier,\
    RandomForestClassifier, ExtremeTreeClassifier
from AutoML.helper.GridParams import GetGridParams
from AutoML.helper.check import check_algorithm_exits

class IntanceModel(object):
    def __init__(self, category, framework, algorithm_name, model_parameters=None, grid_search=True, cv=3):
        self.category = category
        self.framework = framework
        self.algorithm_name = algorithm_name
        self.model_parameters = model_parameters
        self.grid_search = grid_search
        self.cv = cv

    def instance(self):
        if self.grid_search:
            self.model_parameters = GetGridParams(self.category, self.framework, self.algorithm_name)
            # Start to instant grid model
            self._instant_grid_model()


    def _instant_grid_model(self, model_instance, grid_params):
        """
        This function is used to instant grid search model.
        :param model_instance: Already instant model object.
        :param grid_params: Model grid search parameters dierctory.
        :return: Instant Grid search model using given params and cv parameter.
        """
        if model_instance is None:
            raise ModuleNotFoundError('This module is not instanted in this framework! Please Instant it first')
        if grid_params is None:
            raise AttributeError('If want to use Grid_search, must provide grid_params parameter!')
        return GridSearchCV(estimator=model_instance, param_grid=grid_params, cv=self.cv)

    def _instant_model_by_name(self):
        # First to load all implement algorithms from algorithms.json file
        check_algorithm_exits(self.category, self.framework, self.algorithm_name)

        # This is according to different category, framework, algorithm to instance the model object!
        if self.category.lower() == 'classification':
            if self.algorithm_name == 'LogisticRegressionClassifier':
                self.model = LogisticRegressionClassifier(self.model_parameters)




