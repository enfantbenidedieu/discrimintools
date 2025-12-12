# -*- coding: utf-8 -*-
from numpy import sqrt, diag, outer
from pandas import DataFrame

#intern function
from .utils import check_is_squared

def cov_to_corr(X):
    """
    Covariance to Correlation

    Convert covariance matrix to pearson correlation matrix

    Parameters:
    -----------
    X : 2-D array or DataFrame with (n_columns, n_columns)
        Covariance matrix.

    Returns:
    --------
    Y : 2-D array or DataFrame of shape (n_columns, n_columns)
        Correlation matrix.
    """
    #check if X is squared
    check_is_squared(X=X)

    v = sqrt(diag(X))
    outer_v = outer(v, v)
    Y = X / outer_v
    Y[X == 0] = 0

    #convert to DataFrame
    if isinstance(X,DataFrame):
        Y = DataFrame(Y,index=X.columns,columns=X.columns)
    return Y