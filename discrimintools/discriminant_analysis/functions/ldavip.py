# -*- coding: utf-8 -*-
from numpy import linalg, sqrt
from scipy import stats
from pandas import Series, concat
from functools import reduce
from collections import OrderedDict, namedtuple

#intern function
from .utils import check_is_dataframe, check_is_series
from .sscp import sscp
from .powerset import powerset

def ldavip(X,y,level=0.5,all_vars=True):
    """
    Variables Importance for Prediction in Linear Discriminant Analysis (LDAVIP)

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

    y : Series of shape (n_samples,)
        Target values. True labels for ``X``.

    level : float, default=None
        Significance level for the variable importance critical probability. If None 5e-2 is used as the significance level for the variabe importance.

    all_vars : bool, default=True
        If to test all subset of variables.

    Returns
    -------
    NamedTuple :
    
        - vip : DataFrame of shape (n_features, 6) including:
            - "Wilks' Lambda" : Wilks' lambda
            - "Partial R-Square" : Partial R square
            - "F Value" : Fisher approximations
            - "Num DF" : numerator degree of freedom
            - "Den DF" : denominator degree of freedom
            - "Pr>F" : probability values

        - selected : list
            Selected variables
    """
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X)

    #check if y is an instance of class pd.Series
    check_is_series(y)

    #variable importance for prediction
    def importance(ls,Vb,Wb,n_samples,n_classes,lw):
        #convert to list if string
        if isinstance(ls,str):
            to_remove, name = [ls], ls
        elif isinstance(ls,(list,tuple)) and len(ls)==1:
            to_remove, name = [ls[0]], ls[0]
        elif isinstance(ls,(list,tuple)) and len(ls)>1:
            to_remove, name = [x for x in ls], ",".join(ls)
        #number of features and number of remove
        n_features, n_removes = Vb.shape[1], len(to_remove)
        #Lambda of Wilk's without the variable(s)
        lwVar = linalg.det(Wb.drop(index=to_remove,columns=to_remove))/linalg.det(Vb.drop(index=to_remove,columns=to_remove)) if n_removes < n_features else 1
        #A and C
        A, C = n_samples - n_classes - (n_features - n_removes) - 0.5*(n_removes - n_classes + 2), 0.5*(n_removes*(n_classes-1) - 2)
        #B
        B = sqrt(((n_removes**2)*(n_classes - 1)**2 - 4)/(n_removes**2 + (n_classes - 1)**2 - 5)) if (n_removes**2 + (n_classes - 1)**2 - 5) > 0 else 1
        #degree of freedom
        ddl1, ddl2 = n_removes*(n_classes - 1), A*B - C
        #partial rsquared
        partial_rsq = lw/lwVar
        #Fisher statistics
        f_value = ((1 - partial_rsq**(1/B))/(partial_rsq**(1/B)))*(ddl2/ddl1)
        #Fisher pvalue
        p_value = stats.f.sf(f_value,ddl1,ddl2) if f_value >=0 else stats.f.sf(1/f_value,ddl1,ddl2)
        #convert to Series
        return Series([lwVar,partial_rsq,f_value,ddl1,ddl2,p_value],index=["Wilks' Lambda","Partial R-Square","F Value","Num DF","Den DF","Pr>F"],name=name)
    #classes - unique element in y
    classes = sorted(y.unique().tolist())
    #define subset of X
    X_k = {k : X.loc[y[y==k].index,:] for k in classes}
    #number of classes
    n_samples, n_classes = X.shape[0], len(classes)
    #total-sample and within-class SSCP matrix
    tsscp, wsscp = sscp(X), {k: sscp(X_k[k]) for k in classes}
    #pooled within-class SSCP matrix
    pwsscp = reduce(lambda i , j : i + j, wsscp.values())
    #biaised covariance matrices : total -sample and pooled within-class
    tcovb, pwcovb = tsscp.div(n_samples), pwsscp.div(n_samples)
    #global Wilks' Lambda
    lw = linalg.det(pwcovb)/linalg.det(tcovb)
    #variables for which variable importance should be calculated
    vars = list(X.columns)
    if all_vars:
        vars = powerset(vars)
    #variables importance for prediction
    vip = concat((importance(ls=ls,Vb=tcovb,Wb=pwcovb,n_samples=n_samples,n_classes=n_classes,lw=lw).to_frame() for ls in vars),axis=1).T
    #convert to integer
    vip["Num DF"], vip["Den DF"] = vip["Num DF"].astype(int), vip["Den DF"].astype(int)
    #select variables using level
    select_vars = list(vip[vip["Pr>F"]<level].index)
    #convert to ordered dictionary
    res_ = OrderedDict(vip=vip,selected=select_vars)
    #convert to namedtuple
    return namedtuple("LdaVIP",res_.keys())(*res_.values())