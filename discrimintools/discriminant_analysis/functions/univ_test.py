# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from .utils import check_is_dataframe, check_is_series
from .model_matrix import model_matrix
from .eta_sq import eta_sq

def univ_test(X,y):
    """
    Univariate Test Statistics

    Performns univariate test statistics

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        Categorical features are encoded into binary variables.

    y : Series of shape (n_samples,)
        Target values. True labels for ``X``.

    Returns
    -------
    anova : DataFrame of shape (n_features, 8)
        Univariate test statistics.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import univ_test
    >>> DTrain = load_wine()
    >>> yTrain, XTrain = DTrain["Quality"], DTrain.drop(columns=["Quality"])
    >>> anova = univ_test(XTrain,yTrain)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(X)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if y is an instance of class pd.Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_series(y)

    #check if len are equal
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X must be equal to the number of samples in y")
    
    #check if all elements in y are string
    if not all(isinstance(kq, str) for kq in y):
        raise TypeError("All elements in y must be a string")
    
    #encode categorical variables into binary without first level.
    X = model_matrix(X=X)
    
    #squared correlation ratio - eta2
    anova = concat((eta_sq(X[k],y).to_frame(k) for k in list(X.columns)),axis=1).T
    #convert to integer
    anova["Num DF"], anova["Den DF"] = anova["Num DF"].astype(int), anova["Den DF"].astype(int)
    return anova