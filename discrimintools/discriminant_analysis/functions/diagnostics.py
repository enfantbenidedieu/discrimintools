# -*- coding: utf-8 -*-
from scipy import stats
from numpy import linalg, log, sqrt, nan
from pandas import DataFrame

def diagnostics(Vb,Wb,n_samples,n_classes):
    """
    Compute Global statistic - Wilks' Lambda - Bartlett statistic and Rao

    Parameters
    ----------
    V : DataFrame of shape (n_features, n_features).
        Biased total-sample covariance matrix.

    W : DataFrame of shape (n_features, n_features).
        Biased pooled within-class covariance matrix.

    n_samples : int
        Number of samples.

    n_classes : int
        Number of classes.
    
    Returns
    -------
    manova : DataFrame of shape (3, 3)
        Multivariate Analysis of Variance.
    """
    #set number of features
    n_features = Vb.shape[0]
    #Wilks' Lambda
    lw = linalg.det(Wb)/linalg.det(Vb)
    ##Bartlett Test - statistic and degree of freedom
    LB, ddl = -(n_samples - 1 - 0.5*(n_features + n_classes))*log(lw), n_features*(n_classes - 1)
    
    ## RAO test
    #value of A and C
    A, C = n_samples - n_classes - 0.5*(n_features - n_classes + 2), 0.5*(ddl - 2)
    #value of B
    B = sqrt(((n_features**2)*((n_classes - 1)**2)-4)/(n_features**2 + (n_classes - 1)**2 - 5)) if (n_features**2 + (n_classes - 1)**2 - 5) > 0 else 1
    #second degree of freedom and RAO statistic
    ddl2 = A*B-C
    rao = ((1-(lw**(1/B)))/(lw**(1/B)))*(ddl2/ddl)
    
    #convert to DataFrame
    manova = DataFrame({"Statistic" : ["Wilks' Lambda",f"Bartlett -- C({int(ddl)})",f"Rao -- F({int(ddl)},{int(ddl2)})"],
                        "Value" : [lw, LB, rao],
                        "p-value": [nan, stats.chi2.sf(LB,ddl), stats.f.sf(rao,ddl,ddl2)]})
    return manova