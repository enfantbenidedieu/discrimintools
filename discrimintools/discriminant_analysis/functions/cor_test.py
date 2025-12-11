# -*- coding: utf-8 -*-
from pandas import DataFrame

#intern function
from .utils import check_is_dataframe
from .cov_to_cor_test import cov_to_cor_test

def cor_test(X) -> DataFrame:
    """
    Test for Pearson Correlation Between Paired Samples
    ----------------------------------------------------

    Description
    -----------
    Test for association between paired samples, using Pearson's product moment correlation coefficient

    Usage
    -----
    ```python
    >>> cor_test(X)
    ```

    Parameters
    ----------
    X : pandas DataFrame of shape (n_samples, n_features)
        Training data

    Returns
    -------
    cor : a pandas DataFrame of shape (n_features, n_features)
        The pearson correlation coefficient and p-value of the test

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)
    return cov_to_cor_test(X=X.cov(ddof=0),n_samples=X.shape[0])