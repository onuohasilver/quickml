import time
import sys
import pandas as pd
import os
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

class MultiColumnLabelEncoder:
    """
    Parameters
    ----------
    columns : array-like

    Examples
    --------
    dsa_2019 = {'Names': ['Obinna', 'Adeola', 'Hakeem'], 'Gender': ['Male', 'Female', 'Male']}
    dsa_df = pd.DataFrame(data = dsa_2019)
    dsa_df

     	Names 	Gender
    0 	Obinna 	Male
    1 	Adeola 	Female
    2 	Hakeem 	Male

    enc = MultiColumnLabelEncoder(columns=['Gender'])
    enc.fit_transform(dsa_df)

     	Names 	Gender
    0 	Obinna 	1
    1 	Adeola 	0
    2 	Hakeem 	1

    """

    def __init__(self, columns=None):
        # list of column to encode    
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.

        """

        X_copy = X.copy()

        if self.columns is not None:
            for column in self.columns:
                X_copy[column] = LabelEncoder().fit_transform(X_copy[column])
        else:
            for colname, column in X_copy.iteritems():
                X_copy[colname] = LabelEncoder().fit_transform(column)

        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


