# -*- coding: utf-8 -*-
#intern function
from .utils import check_is_dataframe

def describe(X):
    """
    Generate descriptive statistics

    Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding `NaN` values.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Training data, where `n_samples` is the number of features and `n_features` is the number of features.

    Returns
    -------
    stats : DataFrame of shape (n_features, 5)
        Descriptive statistics.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)
    return X.describe().T.assign(count=lambda x : x["count"].astype(int))