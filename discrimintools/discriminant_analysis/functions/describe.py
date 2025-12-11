# -*- coding: utf-8 -*-
from pandas import DataFrame

#intern function
from .utils import check_is_dataframe

def describe(X) -> DataFrame:
    """
    Generate descriptive statistics

    Description
    -----------
    Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding `NaN` values.

    Parameters
    ----------
    X : pandas DataFrame of shape (n_samples, n_features)
        Training data, where `n_samples` is the number of features and `n_features` is the number of features.

    Returns
    -------
    stats : pandas DataFrame of shape (n_features, 5)
        Descriptive statistics

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)
    return X.describe().T.assign(count=lambda x : x["count"].astype(int))