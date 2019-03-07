# -*- coding:utf-8 -*-
"""This is based on sklearn preprocessing module to do machine learning feature engineering process"""
from sklearn import preprocessing
import scipy as sp
import pandas as pd
import numpy as np


class StandardScaler(object):
    """
    This class is used for standard data to be normal distribution.
    Because some machie learning needs the data to be Gaussian distribution(e.g. SVM)
    """
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.standard = preprocessing.StandardScaler(with_mean=self.with_mean, with_std=self.with_std)

    def fit(self, X):
        # Here I also want to support for sparse input data
        if isinstance(X, sp.sparse.csr_matrix) or isinstance(X, sp.sparse.bsr_matrix):
            self.standard.with_mean = False
        self.standard.fit(X)
        return self

    def transform(self, X):
        return self.standard.transform(X)


class MinMaxScaler(object):
    """
    This class is used to make data to be unit scale(0~1)
    """
    def __init__(self):
        self.minmaxscaler = preprocessing.MinMaxScaler()

    def fit(self, X):
        self.minmaxscaler.fit(X)
        return self

    def transform(self, X):
        return self.minmaxscaler.transform(X)


class MaxAbsScaler(object):
    """
    This class is used to make data to be fixed scale(-1~1)
    """
    def __init__(self):
        self.maxabsscaler = preprocessing.MaxAbsScaler()

    def fit(self, X):
        self.maxabsscaler.fit(X)
        return self

    def transform(self, X):
        return self.maxabsscaler.transform(X)


class QuantileScaler(object):
    """
    This class is used to transform data to be unit or Gaussian distribution.
    But in fact, sometimes by using the function, maybe some algorithms may
    get a better evaluation result.
    """
    def __init__(self):
        self.quantilescaler = preprocessing.QuantileTransformer()

    def fit(self, X):
        self.quantilescaler.fit(X)
        return self

    def transform(self, X):
        return self.quantilescaler.transform(X)


class NormizerScaler(object):
    """
    This class is used to make data to be normal distribution.
    By using this function, the data will be really small as normal distribution,
    for some algorithms that are sensitive about data scale, maybe use it will be
    a bad idea.
    """
    def __init__(self, norm='l2'):
        self.norm = norm
        self.norm = preprocessing.Normalizer(norm=self.norm)

    def fit(self, X):
        self.norm.fit(X)
        return self

    def transform(self, X):
        return self.norm.transform(X)

class OneHotScaler(object):
    """
    This class is used to make categorical data to be one-hot for after training.
    """
    def __init__(self, n_values='auto', categorical_features=None, handle_unknown='ignore'):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.handle_unknown = handle_unknown
        self.onehot = preprocessing.OneHotEncoder(n_values=self.n_values,
            categorical_features=self.categorical_features, handle_unknown=self.handle_unknown)

    def fit(self, X):
        self.onehot.fit(X)
        self.categories_ = self.onehot.categories_
        return self

    def transform(self, X):
        return self.onehot.transform(X)

class PolyScaler(object):
    """Poly degree=2 means that (x1, x2) transformed to (1, x1, x2, x1**2, x1*x2, x2**2)"""
    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        # Where or not just to get interaction result, just get(1, x1, x2, x1*x2)
        self.interaction_only = interaction_only
        self.poly = preprocessing.PolynomialFeatures(degree=degree, interaction_only=self.interaction_only)

    def fit(self, X):
        self.poly.fit(X)
        return self

    def transform(self, X):
        return self.poly.transform(X)


class CustomScaler(object):
    """
    This class can be used to use user defined function to make the processing steps.
    """
    def __init__(self, function=None):
        if not hasattr(function, '__call__'):
            raise AttributeError('Provide function value must be funtion type!')
        self.function = function
        self.customscaler = preprocessing.FunctionTransformer(self.function, validate=True)

    def fit(self, X):
        # Do nothing for consistent funtion type
        pass

    def transform(self, X):
        # from sklearn: Please note that a warning is raised and can be turned into an error with a filterwarnings
        import warnings
        warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)
        self.customscaler.transform(X)

class ImputerScaler(object):
    """
    This class is used to impute data for test data based on training data,
    because sometimes test dataset may also contain columns that exist NAN values but
    not contained in training data, so this is to make transform based on strategy to impute
    test datasets.
    """
    def __init__(self, conti_strategy='mean', cate_strategy='most_frequent'):
        """
        :param conti_strategy: Default is 'mean', for now is just can be used for 'mean' or 'sum'
        :param cate_strategy: Default is 'most_frequent', can be added with other strategy.
        """
        self.conti_strategy = conti_strategy
        self.cate_strategy = cate_strategy

    # This is training data to be fitted
    def fit(self, X):
        # X data type must be DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [str(i) for i in range(X.shape[1])]

        self.data_dim = X.shape[1]

        # Get most frequent values for category columns
        def _most(lst):
            # lst must be list type.
            return max(set([x for x in lst if x is not None or x is not np.nan or x != '']), key=lst.count)

        columns_dtypes = np.array(X.dtypes.tolist())
        columns = X.columns

        # Get columns type based on whether data type is 'O'(object)
        conti_columns = [True if x != 'O' else False for x in columns_dtypes]
        cate_columns = [False if x == 'O' else False for x in columns_dtypes]

        conti_columns = columns[conti_columns]
        cate_columns = columns[cate_columns]

        # Here make one Directory for all the columns converting result
        self.stra_values = {}

        # Continous columns
        if len(conti_columns) != 0:
            if self.conti_strategy == 'mean':
                self.stra_values.update(X[conti_columns].mean().to_dict())
            elif self.conti_strategy == 'sum':
                self.stra_values.update(X[conti_columns].sum().to_dict())
            else:
                # TODO add more strategy for continous columns
                pass

        # Category columns
        if len(cate_columns) != 0:
            if self.cate_strategy == 'most_frequent':
                if len(cate_columns) == 1:
                    self.stra_values.update({cate_columns[0]: _most(list(X[cate_columns]))})
                else:
                    c_values = [_most(list(X[cate_columns[i]])) for i in range(len(cate_columns))]
                    for i in range(len(c_values)):
                        self.stra_values.update({cate_columns[i]: c_values[i]})
        else:
            # TODO add more strategy for category columns
            pass

        return self


    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [str(i) for i in range(X.shape[1])]

        if self.data_dim != X.shape[1]:
            raise AttributeError('Test data is not same dimension of Train data. Train is %d, Test is %d'
                                 %(self.data_dim, X.shape[1]))

        # Every thing passed, so just fillna inplace for the original dataset.
        X.fillna(self.stra_values, inplace=True)
        return X


class ImputerScalerTrainingData(object):
    """This class is based on impute data based on training data if some columns that contain some null value,
    but infact that sometimes, test datasets maybe contain some columns that don't contain NAN values,
    so I will make another class to make the missing value based on all the 'strategy' of training data."""
    def __init__(self, conti_strategy='mean', cate_strategy='most_frequent'):
        """
        :param conti_strategy: Default is 'mean', for now is just can be used for 'mean' or 'sum'
        :param cate_strategy: Default is 'most_frequent', can be added with other strategy.
        """
        self.conti_strategy = conti_strategy
        self.cate_strategy = cate_strategy

    def fit(self, X):
        # Because I want to make more robust for missing value, here I will store data information for this scaler
        # Noted: This X must be DataFrame type
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Get most frequent values.
        def _most(lst):
            # lst must be list type.
            return max(set([x for x in lst if x is not None or x is not np.nan or x != '']), key=lst.count)

        # Judge which column contain 'nan'
        self.nan_columns = X.columns[X.isnull().sum() > 0]
        dtype_list = X[self.nan_columns].dtypes.tolist()

        # Big O means object
        conti_bool = [True if x != 'O' else False for x in dtype_list ]
        cati_bool = [True if x == 'O' else False for x in dtype_list ]

        self.conti_cols = self.nan_columns[conti_bool]
        self.cate_cols = self.nan_columns[cati_bool]
        self.conti = {}
        self.cate = {}
        if self.conti_strategy == 'mean':
            for conti_col in self.conti_cols:
                self.conti[conti_col] = np.mean(X[conti_col], axis=0)
        else:
            for conti_col in self.conti_cols:
                self.conti[conti_col] = np.sum(X[conti_col], axis=0)

        if self.cate_strategy == 'most_frequent':
            for cat_col in self.cate_cols:
                print('cat_col', cat_col)
                print(_most(X[cat_col].tolist()))
                self.cate[cat_col] = _most(X[cat_col].tolist())
        else:
            # If get better solution, then change this.
            pass
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        new_nan_cols = X.columns[X.isnull().sum() >0]
        if not np.all(np.isin(new_nan_cols, self.nan_columns)):
            raise ValueError('Transform Data contained some missing columns that not in training data.')

        # Transform data
        if len(self.conti) != 0:
            X[self.conti_cols] = X[self.conti_cols].fillna(self.conti)
        if len(self.cate_cols) != 0:
            X[self.cate_cols] = X[self.cate_cols].fillna(self.cate)
        return X



if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from Accenture.AutoML.Classification.sklearn.classifier import LR

    # x, y = load_iris(return_X_y=True)
    # x = StandardScaler().fit(x).transform(x)
    # lr = LR()
    # print('LR score:', lr.fit(x, y).score(x, y))

    a = np.array([[np.nan, 1, 1, 2, 1], [np.nan, 3, 3, np.nan, 1]]).reshape(-1, 2)
    b = np.array([np.nan, 1, 2, 3, 1]).reshape(-1, 1)
    # d = np.array(['a', 'b', '', 'a', 'a']).reshape(-1, 1)
    c = np.concatenate((a, b), axis=1)
    df = pd.DataFrame(c)


    #print('Original DF:', df)
    imp = ImputerScaler(conti_strategy='mean')
    imp.fit(df)
    # print('New DF:')
    print(imp.transform(df))

