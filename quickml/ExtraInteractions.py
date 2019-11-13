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
from ColumnEncoder import MultiColumnLabelEncoder

class Add_interactions:
    """
    Parameters
    ----------
    columns : array-like

    Examples
    --------
    dsa_2019 = {'Names': ['Obinna', 'Adeola', 'Hakeem'], 'Gender': ['Male', 'Female', 'Male'],
           'Height': [23, 45, 67], 'State':['Abia', 'Ogun', "Ogun"]}
    dsa_df = pd.DataFrame(data = dsa_2019)
    dsa_df

     	Names 	Gender 	Height 	State
    0 	Obinna 	Male 	23 	Abia
    1 	Adeola 	Female 	45 	Ogun
    2 	Hakeem 	Male 	67 	Ogun


    If Column = None i.e No column is specified
    interact = Add_interactions()
    interact.fit_transform(dsa_df)

 	       Names 	Gender 	Height 	State 	Names_Gender 	Names_Height 	Names_State 	Gender_Height 	Gender_State 	Height_State
    0 	    2.0 	 1.0 	 0.0 	 0.0 	   2.0 	         0.0 	            0.0 	       0.0 	          0.0 	            0.0
    1 	    0.0 	 0.0 	 1.0 	 1.0 	   0.0 	         0.0 	            0.0 	       0.0 	          0.0 	            1.0
    2 	    1.0 	 1.0 	 2.0 	 1.0 	   1.0 	         2.0 	            1.0 	       2.0 	          1.0 	            2.0

    If Column = ['Names', 'Gender', 'State']
    interact = Add_interactions(columns=Column)
    interact.fit_transform(dsa_df)

     	Names 	Gender 	Height 	State 	Names_Gender 	Names_State 	Gender_State
    0 	  2 	  1 	 23 	  0 	    2.0 	       0.0 	         0.0
    1 	  0 	  0 	 45 	  1 	    0.0 	       0.0 	         0.0
    2 	  1 	  1 	 67 	  1 	    1.0 	       1.0 	         1.0


    """

    def __init__(self, columns=None):
        # Columns to interact   
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms

        """

        X_copy = X.copy()
        enc = MultiColumnLabelEncoder(columns=self.columns)
        X_copy_enc = enc.fit_transform(X_copy)
        poly = PolynomialFeatures(interaction_only=True, include_bias=False)

        if self.columns is not None:
            column_combos = list(combinations(list(self.columns), 2))
            X_columns = X_copy_enc.loc[:, self.columns]
            new_col = list(X_columns.columns)
            colnames = new_col + ['_'.join(x) for x in column_combos]

            df_new = poly.fit_transform(X_columns)
            dd = pd.DataFrame(df_new, columns=colnames)
            data = pd.concat([X_copy_enc, dd], axis=1)
            data = data.loc[:, ~data.columns.duplicated()]

        else:
            column_combos = list(combinations(list(X_copy.columns), 2))
            colnames = list(X_copy.columns) + ['_'.join(x) for x in column_combos]
            df_new = poly.fit_transform(X_copy_enc)

            data = pd.DataFrame(df_new, columns=colnames)

        return data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)