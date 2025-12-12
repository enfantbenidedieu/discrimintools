# -*- coding: utf-8 -*-
#intern function
from .utils import check_is_dataframe

def sscp(X):
    """
    Sum of Square Cross Product

    Performs sum of square cross product

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of features and ``n_features`` is the number of features.

    Returns
    -------
    sscp : DataFrame of shape (n_features, n_features)
        Sum of Square Cross Product.
    """
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)

    #centered the dataset
    X = X.sub(X.mean(axis=0),axis=1)
    #sum of squares
    return X.T.dot(X)