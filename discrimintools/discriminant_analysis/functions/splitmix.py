# -*- coding: utf-8 -*-
from numpy import number
from pandas import DataFrame, Series, Categorical, concat
from collections import namedtuple

def splitmix(X):
    """
    Split Mixed Data

    Splits a mixed data matrix in two data sets: one with the quantitative variables and one with the qualitative variables.

    Parameters
    ----------
    X : Dataframe of shape (n_samples, n_columns)
        Input data.

    Return
    ------
    NamedTuple:

        - quanti : None or a DataFrame of shape (n_samples, n_quanti)
            Quantitative variables.

        - quali : None or a DataFrame of shape (n_samples, n_quali)
            Qualitative variables.

        - n : int
            The number of rows.

        - k1 : int
            Number of quantitative variables.

        - k2 : int
            Number of qualitative variables.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    #initialisation
    X_quali, X_quanti, n_quali, n_quanti = None, None, 0, 0

    #select object or category
    is_quali = X.select_dtypes(include=["object","category"])
    if not is_quali.empty:
        X_quali = concat((Series(Categorical(is_quali[q],categories=sorted(is_quali[q].dropna().unique().tolist()),ordered=True),index=is_quali.index,name=q) for q in is_quali.columns),axis=1)
        if isinstance(X_quali, Series):
            X_quali = X_quali.to_frame()
        n_quali = X_quali.shape[1]
    
    #select all numerics columns
    is_quanti = X.select_dtypes(include=number)
    if not is_quanti.empty:
        X_quanti = concat((is_quanti[k].astype(float) for k in is_quanti.columns),axis=1)
        if isinstance(X_quanti, Series):
            X_quanti = X_quanti.to_frame()
        n_quanti = X_quanti.shape[1]

    #convert to namedtuple
    return namedtuple("SplitmixResult",["quanti","quali","n","k1","k2"])(X_quanti,X_quali,X.shape[0],n_quanti, n_quali)