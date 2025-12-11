# -*- coding: utf-8 -*-
from pandas import concat, DataFrame, Series

def concat_empty(
        initial,actual,axis=0
) -> DataFrame|Series:
    """
    Concatenate DataFrame or Series

    Description
    -----------
    Concatenate pandas objects along a particular axis.

    Parameters
    ----------
    initial : Series or DataFrame
        Initial objects.

    actual : Series or DataFrame
        actual objects.
    
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.

    Returns
    -------
    obj : DataFrame or Series
        Concatenate object

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    obj = actual if initial is None else concat((initial,actual),axis=axis)
    return obj                                      