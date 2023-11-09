'''
Custom-made feature engineering functions, compatible with sklearn Pipeline
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MissingBinaryTransformer(BaseEstimator, TransformerMixin):
    '''Function transforms missing variables to 0 and remaining to 1'''
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    # Trivial method to be compatible with Sklearn pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
            X[feature] = np.where(X[feature] == 'Missing', 0, 1)

        return X
    

class GenderBinaryTransformer(BaseEstimator, TransformerMixin):
    '''Function that converts female to 0 and male to 1'''
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    # Trivial method to be compatible with Sklearn pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature] == 'female', 0, 1)

        return X


class NonZeroTransformer(BaseEstimator, TransformerMixin):
    '''
    Function that if the cell value is less than 0.01,
    Makes it 0.01 to prevent problems while log transformation
    For instance, for Fare column
    '''
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    # Trivial method to be compatible with Sklearn pipeline
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(int(0))
            X[feature] = np.where(X[feature] < 0.1, 0.1, X[feature])

        return X