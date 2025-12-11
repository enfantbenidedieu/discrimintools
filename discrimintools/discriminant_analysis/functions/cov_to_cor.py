# -*- coding: utf-8 -*-
from numpy import sqrt, diag, outer
from pandas import DataFrame

#intern function
from .utils import check_is_squared

def cov_to_corr(X):
    """
    Covariance to Correlation
    -------------------------

    Description
    -----------
    Convert covariance matrix to pearson correlation matrix

    Parameters:
    -----------
    `X`: a numpy 2-D array or a pandas DataFrame with (n_columns, n_columns)

    Returns:
    --------
    a numpy 2-D array or a pandas DataFrame of shape (n_columns, n_columns)
    
    Authors:
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
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