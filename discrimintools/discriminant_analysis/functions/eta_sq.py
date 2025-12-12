# -*- coding: utf-8 -*-
from numpy import array, unique, sum, mean, where
from scipy import stats
from pandas import Series

def eta_sq(x,y):
    """
    Eta Squared

    Performs eta squared which represent an estimate of how much variance in the response variables (x) is accounted for by the explanatory variable (y)

    Parameters
    ----------
    x : 1-D array or Series of shape (n_samples,).
        x contains quantitative variable.

    y : 1-D array or Series of shape (n_samples,).
        y contains qualitative variables.

    Returns
    -------
    Series
    """
    #convert to array
    x, y = array(x), array(y)
    #set classes
    classes = sorted(list(unique(y)))
    #number of classes
    n_samples, n_classes, i_k =  len(x), len(classes), {k : where(y == k)[0] for k in classes}
    #subset of x and number of elements
    x_k, n_k = {k : x[i_k[k]] for k in classes}, {k : len(i_k[k])  for k in classes}
    #within and between sum of squared
    ssw, ssb = sum([sum((x_k[k] - mean(x_k[k]))**2) for k in classes]), sum([n_k[k]*(mean(x_k[k]) - mean(x))**2 for k in classes])
    #eta squared
    sqeta = ssb/(ssw + ssb)
    #r-square/(1 - rsquare)
    partiel_sqeta = sqeta/(1 - sqeta)
    #degree of freedom
    ddl1, ddl2 = n_classes - 1, n_samples - n_classes
    #F Value
    f_value = (ssb/ssw)*(ddl2/ddl1)
    #P Value
    p_value = stats.f.sf(f_value, ddl1, ddl2)
    #convert to pandas Series
    return Series([ssw,ssb,sqeta,partiel_sqeta,f_value,ddl1,ddl2,p_value],index=["Within SS","Between SS","R-Square","R-Square/(1-RSq)","F Value","Num DF","Den DF","Pr>F"])