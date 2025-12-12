# -*- coding: utf-8 -*-
from numpy import zeros,diag, array, sqrt
from pandas import Series
from collections import OrderedDict, namedtuple
from scipy import linalg

def plsrvip(
        obj, threshold=1.0
):
    """
    Variable Importance in Projection for Partial Least-Squares Regression

    Calculate variable importance in projection (VIP) scores for a partial least-squares (PLS) regression model. You can use VIP to select predictor variables when multicollinearity exists among variables. 
    Variables with a VIP score greater than 1 are considered important for the projection of the PLS regression model (see https://doi.org/10.1016/j.chemolab.2004.12.011)
    
    Parameters
    ----------
    obj : plsr function
        Object from `PLSRegression` function.

    threshold : float, default = 1.
        Threshold below which the variable is rejected

    Returns
    -------
    NamedTuple:

        - vip : Series of shape (n_features,)
            Variable importance in projection for partial least-squares regression

        - selected : list
            Selected variables
        
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object from 'PLSRegression' function
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.__class__.__name__ != "PLSRegression":
        raise TypeError("'obj' must be an instance of class PLSRegression.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if threshold is not None
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if threshold is None:
        threshold = 1.0
    elif not isinstance(threshold,float):
        raise TypeError("{} is not supported".format(type(threshold)))
    elif threshold < 0 :
        raise ValueError("the 'threshold' value must be positive.")

    #extract elements
    t, w, q = obj.x_scores_, obj.x_weights_, obj.y_loadings_
    #weights shape
    p, h = w.shape
    #r-square - square of pearson correlation
    s = diag(t.T @ t @ q.T @ q).reshape(h, -1)
    #sum of r-square
    total_s = sum(s)
    vip = zeros((p,))
    for i in range(p):
        weight = array([ (w[i,j] / linalg.norm(w[:,j]))**2 for j in range(h)])
        vip[i] = sqrt(p*(s.T @ weight)/total_s)
    #convert to Series
    vip = Series(vip,index=obj.feature_names_in_,name="VIP")
    #select variables using threshold
    select_vars = list(vip[vip > threshold].index)
    #if at most one variable, select the two first
    if len(select_vars) < 2:
        select_vars  = list(vip.sort_values(ascending=False).index[:2])
    #convert to ordered dictionary
    res_ = OrderedDict(vip=vip,selected=select_vars)
    #convert to namedtuple
    return namedtuple("PLSRVIP",res_.keys())(*res_.values())