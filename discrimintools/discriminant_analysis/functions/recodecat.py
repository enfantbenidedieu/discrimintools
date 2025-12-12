# -*- coding: utf-8 -*-
from pandas import Series, DataFrame, concat, get_dummies
from collections import namedtuple

#intern function
from .revalue import revalue

def recodecat(X,dummy_na=False):
    """
    Recoding of the categoricals variables

    Recoding of the categoricals variables

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        X contains categoricals variables

    dummy_na : bool.
        add a column to indicate NaNs, if False NaNs are ignored.

    Returns
    -------
    NamedTuple:

        - X : DataFrame of shape (n_samples, n_columns)
            Categorical data.

        - dummies : DataFrame of shape (n_samples, n_categories)
            Disjunctive table.
    """
    if isinstance(X,Series): #if pandas Series, convert to pandas DataFrame
        X = X.to_frame()
    
    if not isinstance(X,DataFrame): #check if X is an instance of class pd.DataFrame
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    X = X.select_dtypes(include=["object","category"]) #select object of category
    if X.empty:
        raise TypeError("All variables in X must be either object or category")

    #revalue
    X = revalue(X)
    #disjunctive table
    dummies = concat((get_dummies(X[q],dtype=int,dummy_na=dummy_na) for q in X.columns),axis=1)
    return namedtuple("recodecat",["X","dummies"])(X,dummies)