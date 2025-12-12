# -*- coding: utf-8 -*-
from scipy import stats
from numpy import prod, sqrt, log
from pandas import DataFrame

def lrtest(n_samples,n_features,n_classes,eigen):
    """
    Likelihood ratio test

    Performs likelihood ratio test.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_features : int
        Number of features.

    n_classes : int
        Number of classes.

    eigen : int, list
        Eigen values.
    
    Returns
    -------
    res : DataFrame of shape (1, 8)
        Likelihood ratio test.
    """
    #statistique de test
    q = 1 if isinstance(eigen,(float,int)) else len(eigen)
    #likelihood ratio value
    lr = prod([(1-i) for i in eigen])
    #r
    r = n_samples - 1 - 0.5*(n_features + n_classes)
    #t
    t = sqrt((q**2*(n_features - n_classes + q + 1)**2-4)/((n_features - n_classes + q + 1)**2 + q**2 - 5)) if ((n_features - n_classes + q + 1)**2 + q**2 - 5) > 0 else 1
    # u
    u = (q*(n_features - n_classes + q + 1)-2)/4
    #degree of freedom
    ddl1, ddl2 = q*(n_features - n_classes + q + 1), r*t - 2*u
    #F de RAO
    f_value = ((1 - lr**(1/t))/(lr**(1/t))) * (ddl2/ddl1)
    #pvalue
    p_fvalue = stats.f.sf(f_value,ddl1,ddl2)
    #
    chi2 = -r*log(lr)
    #Bartlette
    p_chi2 = stats.chi2.sf(chi2,ddl1)
    #columns
    columns = ["Likelihood Ratio","Approximate F value","Num DF","Den DF","Pr>F","Chi-Square","DF","Pr>Chi2"]
    return DataFrame([[lr,f_value,ddl1,ddl2,p_fvalue,chi2,ddl1,p_chi2]], columns=columns)