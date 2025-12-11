# -*- coding: utf-8 -*-
from numpy import array, ndarray, ones, average, cov, sqrt
from pandas import DataFrame,Series,concat
from typing import NamedTuple
from collections import namedtuple

#interns functions
from .utils import check_is_dataframe

def recodecont(X,weights=None) -> NamedTuple:
    """
    Recoding of the continuous data
    -------------------------------

    Description
    -----------
    Recoding of the continuous data

    Usage
    -----
    ```python
    >>> from discrimintools import recodecont
    >>> recodcont = recodecont(X)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame/Series of continuous variables

    `weights`: an optional individuals weights (by default, 1/(number of active individuals) for uniform individuals weights); the weights are given only for the active individuals

    Return
    ------
    namedtuple of pandas DataFrame/Series containing:

    `X`: the continuous DataFrame X with missing values replaced with the column mean values,
    
    `Z`: the standardizd continuous DataFrame,
    
    `center`: the mean value for each columns in X,
    
    `scale`: the standard deviation for each columns of X.
 
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    if isinstance(X,Series): #convert to pandas DataFrame if pandas Series
        X = X.to_frame()

    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)

    #set number of rows and number of columns
    n_rows, n_cols = X.shape

    #set weights
    if weights is None:
        weights = ones(n_rows)/n_rows
    elif not isinstance(weights,(list,tuple,ndarray,Series)):
        raise TypeError("'weights' must be a list or a tuple or a 1-D numpy array or a pandas Series of individuals weights.")
    elif len(weights) != n_rows:
        raise ValueError(f"'weights' must be a list or a tuple or 1-D numpy array or a pandas Series with length {n_rows}.")
    else:
        weights = array([x/sum(weights) for x in weights])
    
    #convert to Series
    weights = Series(weights,index=X.index,name="weight")

    #exclude object of category
    X = X.select_dtypes(exclude=["object","category"])
    if X.empty:
        raise TypeError("All variables in X must be numerics")
    else:
        X = concat((X[k].astype("float") for k in X.columns),axis=1)

    #fill NA by mean
    for k in X.columns:
        if X.loc[:,k].isnull().any():
            X.loc[:,k] = X.loc[:,k].fillna(X.loc[:,k].mean())
    
    if n_rows == 1:
        Z, center, scale = None, X, None
    else:
        #compute weighted average and standard deviation
        center = Series(average(X,axis=0, weights=weights),index=X.columns,name="center")
        scale = Series([sqrt(cov(X.iloc[:,k],aweights=weights,ddof=0)) for k in range(n_cols)],index=X.columns,name="scale")
        #standardization: Z = (X - mu)/sigma
        Z = X.sub(center,axis=1).div(scale,axis=1)
    return namedtuple("recodecont",["X","Z","center","scale"])(X,Z,center,scale)