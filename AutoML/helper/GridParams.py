# -*- coding:utf-8 -*-
import numpy as np
import json
from AutoML.helper.FileLoader import get_file_path

class GetGridParams:
    def __init__(self, category, framework, algorithm_name):
        self.category = category
        self.framework = framework
        self.algorithm_name = algorithm_name

    def get_grid_param(self):
        if self.category == 'classification':
            self._get_cl_params()
        elif self.category == 'regression':
            pass
        elif self.category == 'clustering':
            pass
        elif self.category == 'recommendation':
            pass
        else:
            raise AttributeError('Not get wanted parameter, supported is: classification, regression',
                                 'clustering, recommendation')

        # Now return the predefined grid parameters
        return self.grid_params

    def _check_params(self, category, framework, algorithm_name):
        """
        Here I predefined some algorithms that in 'algorithms.json'. If desired algorithm not in
        predefined, then raise an error.
        :param framework: which framework chosen.
        :param algorithm_name: Which algorithm to use
        :return: Whether give algorithm in existing framework.
        """
        defined_framework = ['sklearn', 'pyspark', 'tensorflow']
        # Load the predefined algorithms from Json file
        try:
            file_path = get_file_path('algorithms.json')
            with open(file_path) as f:
                algo_dire = json.load(f)
                # print('Now get param json result:', algo_dire)
        except IOError as e:
            msg = 'Can get algorithm json from project path' + e
            raise Exception(msg)

        if framework.lower() not in defined_framework:
            raise AttributeError('Wanted framework is not in defined framework: %s'%(str(defined_framework)))

        if algorithm_name not in algo_dire[category][framework.lower()]:
            raise AttributeError('Wanted algorithm %s is not in defined framework: %s'%(algorithm_name,
                                                                                       str(algo_dire[framework])))
        return True


    def _get_cl_params(self):
        """
        This is based on different framework and algorithms to give different grid parameters
        :param framework: Which framework chosen('sklearn', 'pyspark', 'tensorflow')
        :param algogrithm_name: different algorithm name that for different framework
        :return: Nothing but gives self.grid_params value.
        """
        category = 'classification'
        framework = self.framework.lower()
        algorithm_name = self.algorithm_name
        # First to check algorithms
        self._check_params(category, framework, algorithm_name)

        if framework == 'sklearn':
            if algorithm_name == 'LogisticRegressionClassifier':
                self.grid_params = {'C': np.arange(0.01, 10.0, 0.5),
                                    'fit_intercept': [True, False],
                                    'random_state':[1234]}
            elif algorithm_name == 'RandomForestClassifier':
                self.grid_params = {'n_estimators': np.arange(10, 200, 20),
                                    'random_state':[1234]}
            elif algorithm_name == 'SupportVectorMachineClassifier':
                self.grid_params = {'C': np.arange(0.01, 20.0, .5),
                                    'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
                                    'random_state': [1234]}
            elif algorithm_name == 'GradientBoostingTreeClassifier':
                self.grid_params = {'n_estimators': np.arange(10, 300, 20),
                                    'max_depth': np.arange(3, 16, 2),
                                    'random_state': [1234]}
            elif algorithm_name == 'ExtremeTreeClassifier':
                self.grid_params = {'bootstrap': [True, False],
                                    'max_depth': np.arange(3, 16, 2),
                                    'n_estimators': np.arange(10, 200, 20),
                                    'random_state': [1234]}
            elif algorithm_name == 'AdaboostClassifier':
                self.grid_params = {'n_estimators': np.arange(30, 200, 20),
                                    'random_state': [1234]}
            else:
                raise NotImplementedError('For now, this framework %s for'
                                          ' this algorithm %s is not implemented'%(framework, algorithm_name))
                # Can add more predefined algorithms parameters
                pass
        # TODO: Add more framewor here
        elif framework == 'pyspark':
            pass
        elif framework == 'tensorflow':
            pass
        else:
            raise NotImplementedError('For now, this framework %s is not implemented'%(framework))


    def _get_re_params(self):
        """
        This is based on different framework and algorithms to give different grid parameters
        :param framework: Which framework chosen('sklearn', 'pyspark', 'tensorflow')
        :param algogrithm_name: different algorithm name that for different framework
        :return: Nothing but gives self.grid_params value.
        """
        category = 'regression'
        framework = self.framework.lower()
        algorithm_name = self.algorithm_name
        # First to check algorithms
        self._check_params(category, framework, algorithm_name)

        if framework == 'sklearn':
            pass
        elif framework == 'pyspark':
            pass
        elif framework == 'tensorflow':
            pass
        else:
            raise NotImplementedError('For now, this framework %s is not implemented' % (framework))

    def _get_cluster_params(self):
        """
        This is based on different framework and algorithms to give different grid parameters
        :param framework: Which framework chosen('sklearn', 'pyspark', 'tensorflow')
        :param algogrithm_name: different algorithm name that for different framework
        :return: Nothing but gives self.grid_params value.
        """
        category = 'cluster'
        framework = self.framework.lower()
        algorithm_name = self.algorithm_name
        # First to check algorithms
        self._check_params(category, framework, algorithm_name)

        if framework == 'sklearn':
            pass
        elif framework == 'pyspark':
            pass
        elif framework == 'tensorflow':
            pass
        else:
            raise NotImplementedError('For now, this framework %s is not implemented' % (framework))

    def _get_rec_params(self):
        """
        This is based on different framework and algorithms to give different grid parameters
        :param framework: Which framework chosen('sklearn', 'pyspark', 'tensorflow')
        :param algogrithm_name: different algorithm name that for different framework
        :return: Nothing but gives self.grid_params value.
        """
        category = 'recommendation'
        framework = self.framework
        algorithm_name = self.algorithm_name
        # First to check algorithms
        self._check_params(category, framework, algorithm_name)

        if framework == 'sklearn':
            pass
        elif framework == 'pyspark':
            pass
        elif framework == 'tensorflow':
            pass
        else:
            raise NotImplementedError('For now, this framework %s is not implemented' % (framework))


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import load_iris
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC

    x, y = load_iris(return_X_y=True)

    algorithms_dir = {'LogisticRegressionClassifier':LogisticRegression(),
                      'SupportVectorMachineClassifier': SVC(),
                      'GradientBoostingTreeClassifier': GradientBoostingClassifier(),
                      'RandomForestClassifier': RandomForestClassifier(),
                      'AdaboostClassifier': AdaBoostClassifier(),
                      'ExtremeTreeClassifier': ExtraTreesClassifier()}

    # Loop for all algorithms to check whether or not all algorithms have been set properly
    for al in algorithms_dir.keys():
        param_o = GetGridParams('classification', 'sklearn', al).get_grid_param()
        print('For algorithm {}, get parameters: {}'.format(al, param_o))

        grid = GridSearchCV(estimator=algorithms_dir[al], param_grid=param_o, cv=2)
        grid.fit(x, y )
        print('For algorithm {0:s}, evaluate accuracy :{1:.4f}'.format(al, grid.score(x, y)))



