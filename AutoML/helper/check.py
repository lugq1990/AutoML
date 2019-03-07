# -*- coding:utf-8 -*-
from AutoML.helper.FileLoader import get_predefined_algorithms
import numpy as np

def check_truth_pred_shape(x_truth_shape, x_pred_shape):
    """
    To check training data sample shape equals to prediction value shape
    :param x_truth:
    :param x_pred:
    :return:
    """
    if x_truth_shape != x_pred_shape:
        raise ValueError("Prediction Value MUST be same as training values, "
                         "training value shape %s, got shape %s" % (x_truth_shape, x_pred_shape))

def check_label(y):
    """
    To check whether or not label is provided for using model evaluation using labels
    :param y: Truth label object
    :return: Boolean for label provided
    """
    if len(y) == 0:
        raise ValueError('For model evaluation, label must provided!')

def check_params(params):
    """
    This is used to check whether or not the pamameters is given.
    :param params: tuning parameters directory
    :return: Boolean for chosen parameters to tune.
    """
    if params is None:
        return False
    else: return True


def check_algorithm_exits(category, framework, algorithm_name):
    """
    This function is used to check whether or not the algorithm name is in predefine algorithm.json.
    :param algorithm_name: Which algorithm to be checked
    :return: True, otherwise raise an error
    """
    predefined_algorithms = get_predefined_algorithms()
    if category not in predefined_algorithms.keys():
        raise NotImplementedError('This category %s is not implemented now!'%(category))
    if framework not in predefined_algorithms[category].keys():
        raise NotImplementedError('This framework for this % category is '
                                  'not implemented for %s framework'%(category, framework))
    if algorithm_name not in predefined_algorithms[category][framework].keys():
        raise NotImplementedError('This algorithm %s is not '
                                  'implement for %s category of %s framework!'%(algorithm_name,
                                                                                category,
                                                                                framework))
    return True

def check_data_and_label(data, label):
    if isinstance(data, list): np.array(data)
    if isinstance(label, list): np.array(label)

    if data.shape[0] != label.shape[0]:
        raise AttributeError('Given data and label are not same dimension,'
                             ' data row: %d, label row: %d'%(data.shape[0], label.shape[0]))
    return True