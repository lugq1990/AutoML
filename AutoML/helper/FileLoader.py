# -*- coding:utf-8 -*-
"""This module is used to get file path or get file content, now that this file is called FileLoader,
so all file loader result will be implemented in this .py """
import os
import json


def get_base_path():
    """This is used to get project file path, if directory path is: grandfatherpath/father/t.py
    then reture is grandfatherpath/father"""

    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

def get_file_path(*path_components):
    """
    This is used to get the file path
    :param path_components: (list(string)): component of path
    :return: return file path for given file. eg: [dir, file_name] return is dir/file_name
    """
    return os.path.join(get_base_path(), *path_components)

def get_predefined_algorithms(algorithm_path=get_file_path('algorithms.json')):
    with open(algorithm_path) as f:
        return json.load(f)



