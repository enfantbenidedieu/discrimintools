# -*- coding: utf-8 -*-
from numpy import linalg, log
from pandas import Series

#intern function
from .utils import check_is_squared

def cov_infos(X):
    """
    Covariance Matrix Informations
    ------------------------------

    Description
    -----------
    Performs rank and natural logarithm of the covariance matrix.

    Usage
    -----
    ```python
    >>> cov_infos(X)
    ```

    Parameters
    ----------
    `X`: a numpy 2-D array and a pandas DataFrame of shape (n_columns, n_columns)
        covariance matrix

    Returns
    -------
    `infos`: a pandas Series of shape (2,)
        Covariance matrix information: rank and natural log of the determinant

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #check if X is squared
    check_is_squared(X=X)
    log_det = log(linalg.det(X)) if linalg.det(X) > 0 else 0
    infos = Series([linalg.matrix_rank(X),log_det],index=["Rank","Natural Log of the Determinant"])
    return infos