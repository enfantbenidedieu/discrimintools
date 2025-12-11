# -*- coding: utf-8 -*-
from numpy import average, sqrt

def wcorrcoef(x,y,w):
    """
    Weighted Pearson correlation
    ----------------------------

    Parameters
    ----------
    `x`: a numpy 1-D array or a pandas Series

    `y`: a numpy 1-D array or a pandas Series

    `y`: a numpy 1-D array or a pandas Series

    Returns
    -------
    `value`: weighted pearson correlation coefficient
    
    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    def wcov(x, y, w):
        return average((x - average(x, weights=w)) * (y - average(y, weights=w)), weights=w)
    def wcorr(x, y, w):
        return wcov(x, y, w) / sqrt(wcov(x, x, w) * wcov(y, y, w))
    value = wcorr(x,y,w)
    return value