# -*- coding: utf-8 -*-
from numpy import diag, dot
from pandas import DataFrame, concat, Series
from itertools import combinations
from collections import OrderedDict, namedtuple
from scipy.spatial.distance import mahalanobis

#intern function
from .utils import check_is_dataframe, check_is_series

def sqmahalanobis(X,VI,mu=None) -> DataFrame:
    """
    Squared Mahalanobis Distance
    ----------------------------

    Description
    -----------
    Performs the squared mahalanobis distance

    Usage
    -----
    ```python
    >>> sqmahalanobis(X,VI,mu)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_samples, n_features)
        Inout data, where `n_samples` is the number of samples and `n_features` is the number of features.

    `VI`: a numpy 2-D array or a pandas DataFrame of shape (n_features, n_features)
        The inverse of the covariance matrix.

    `mu`: None or a numpy 1-D array or a pandas Series of shape (n_features,):
        average

    Returns
    -------
    `dist2`: a pandas DataFrame of shape (n_samples, n_samples) or (n_samples, ).
        Squared Mahalanobis distance

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)

    #squared mahalanobis distance to origin
    if mu is not None:
        #check if mu is an instance of class pd.Series
        check_is_series(X=mu)
        #center X
        X = X.sub(mu.values,axis=1)
        #squared mahalanobis distance
        dist2 = Series(diag(dot(dot(X,VI),X.T)),index=X.index,name="Sq. Mahal. Dist.")
    #squared mahalanobis distance between paired of index
    else:
        #all combinaisons of two index
        all_index = combinations(X.index,r=2)
        #mahalanobis squared distance between paired of index
        dist2 = DataFrame()
        for i, idx in enumerate(all_index):
            #squared mahalanobis distance
            sqmahal = mahalanobis(u=X.loc[idx[0],:],v=X.loc[idx[1],:],VI=VI)**2
            #convert to DataFrame
            row_dist = DataFrame(OrderedDict(From=idx[0],To=idx[1],dist2=sqmahal),index=[i])
            #concatenate
            dist2 = concat((dist2,row_dist),axis=0,ignore_index=True)
        #rename
        dist2 = dist2.rename(columns={"dist2": "Sq. Mahal. Dist."})
    return dist2

def sqmahalanobistest(X,n_features,n_k):
    """
    Mahalanobis Distance Test
    -------------------------

    Description
    -----------
    Performs the mahalanobis distance test

    Usage
    -----
    ```python
    >>> distmahalanobistest(X,n_features,n_k)
    ```

    Parameters
    ----------
    `X`: a pandas DataFrame of shape (n_classes, n_classes)
        Inout data, where `n_samples` is the number of samples and `n_features` is the number of features.

    `n_features`: int
        Number of features.

    `n_k`: pandas Series
        Number of elements per class.

    Returns
    -------
    `test`: float
        Mahalanobis distance test

    Authors
    -------
    Duvérier DJIFACK ZEBAZE
    """
    f_stat = X.copy()
    n_samples= n_k.sum()
    for i in X.index:
        for j in X.columns:
            #f_stat.loc[i,j] = (n_k[i]*n_k[j])/(n_k[i]+n_k[j])*((n_k[i]+n_k[j] - n_features - 1)/(n_features*(n_k[i]+n_k[j] - X.shape[1]))) * X.loc[i,j]
            f_stat.loc[i,j] = ((n_samples - n_features - 1)/(n_features*(n_samples - X.shape[1]))) * X.loc[i,j]
    print(f_stat)
    res = OrderedDict(mahaldist2=X,statistics=f_stat)
    return namedtuple("distmahalanobistest",res.keys())(*res.values())