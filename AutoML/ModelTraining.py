# -*- coding:utf-8 -*-
"""This is the main model training module that can get REST info to start model training.
For NOW, I just use the GridSearch algorithm to train model to get best parameters, also with
cross-validation."""

from flask import Flask
import json
from sklearn.model_selection import GridSearchCV
# from AutoML.Classification.sklearn.LogisticRegression import LogisticRegression
# from AutoML.Classification.sklearn.RandomForestClassifier import RandomForestClassifier
from AutoML.helper.FileLoader import get_file_path
from AutoML.helper import GridParams



class ModelTraining:
    def __init__(self, category, framework, algorithm_name, model_params=None, grid_search=True, cv=3):
        self.category = category
        self.framework = framework
        self.algorithm_name = algorithm_name
        self.model_params = model_params
        self.grid_search = grid_search
        self.cv = cv
        self.model = self._instante_model()

    # def _get_params(self):
    #     param_path = get_file_path('params.json')
    #     try:
    #         with open(param_path) as f:
    #             return json.load(f)
    #     except IOError as e:
    #         msg = 'When loading file, cause LOAD error' + e
    #         raise Exception(msg)


    def _instante_model(self):
        """
        This function is based on already given category, framework, model_name to instant the model instance
        for model training. If grid_search is True, then here will use predefined parameters to instant Grid_model,
        this is also according to different category, framework, algorithm name, if not grid search, then just
        instant the model based on parameters.
        :return: Instanced model
        """
        if self.grid_search:
            # If use grid to build model, even if given model initilization parameters, just overwrite it.
            self.model_params = GridParams(self.category, self.framework, self.algorithm_name).get_grid_param()
            # According the framework,

        # Get which model to be used
        # if self.model_name == 'LogisticRegression':
        #     self.model = LogisticRegression(self.model_params)
        # elif self.model_name == 'RandomForestClassifier':
        #     self.model = RandomForestClassifier(self.model_params)
        # else:
        #     pass

    def _instante_grid_model(self):
        grid_params = self._get_params()

        if self.model_name == 'LogisticRegression':
            grid_p = grid_params['classification']['sklearn']['LogisticRegression']
            print('grid_p is ', grid_p)
            self.model = GridSearchCV(estimator=self.model, param_grid=grid_p, cv=self.cv)
        else:
            pass



    def train(self, X, y):
        if self.grid_search:
            # If model training start with grid search, then parameters should be loaded from disk
            # TODO: add something for better represent grid search
            self.model_params = self.grid_params
            grid = GridSearchCV(estimator=self.model, param_grid=self.model_params, cv=self.cv)
            grid.fit(X, y)
            self.model = grid.best_estimator_
        else:
            self.model.fit(X, y)

        return 'Finished!'



if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)


    m = ModelTraining(model_name='LogisticRegression')
    # print(m.train(X, y))

