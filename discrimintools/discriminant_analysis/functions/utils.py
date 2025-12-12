# -*- coding: utf-8 -*-
from pandas import DataFrame, Series

def check_is_dataframe(
        X
):
    """
    Performs is_dataframe validation

    Check if X is an instance of class pd.DataFrame

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data for which check should be done
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame. For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    else:
        pass

def check_is_series(
        X
):
    """
    Performs is_series validation

    Check if X is an instance of class pd.Series

    Parameters
    ----------
    X : pandas Series of shape (n_samples,)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an instance of class pd.Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,Series):
        raise TypeError(f"{type(X)} is not supported. Please convert to a Series with pd.Series. For more information see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html")
    else:
        pass

def check_is_bool(
        x
):
    """
    Performs is_bool validation

    Parameters
    ----------
    x : bool
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if an argument is a bool
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(x,bool):
        raise TypeError(f"{type(x)} is not supported")
    else:
        pass

def check_is_squared(
        X
):
    """
    Performs is_squared validation

    Checks if X is fitted by verifying the equality of its dimensions
    
    Parameters
    ----------
    X : 2-D array-like or pandas DataFrame of shape (n_samples, n_columns)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is squared
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if X.shape[0] != X.shape[1]:
        raise TypeError("X must be a squared numpy 2-D array and a squared pandas DataFrame")
    else:
        pass