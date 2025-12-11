# -*- coding: utf-8 -*-
from numpy import sqrt, where
from scipy import stats
from pandas import DataFrame, concat
from collections import OrderedDict
from itertools import combinations

#intern functions
from .cov_to_cor import cov_to_corr
from .utils import check_is_dataframe

def cov_to_cor_test(
        X,n_samples
):
    """
    Covariance to Correlation Test

    Parameters
    ----------
    X : DataFrame of shape (n_features, n_features)
        Covariance matrix.

    n_samples : int 
        The number of samples.

    Returns
    -------
    cor : DataFrame of shape (C_{n_features}^{2},7)
        Pearson correlation test.
    """
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)
    
    def pearsonr(X,var1,var2,n_samples):
        #pearson correlation and degree of freedom
        cor, dof = X.loc[var1,var2], n_samples - 2
        #t value
        t_value = cor*sqrt((dof/(1- cor**2))) if (1- cor**2) != 0 else 0
        #p value
        p_value = 2*stats.t.sf(abs(t_value), dof)
        #convert to ordered dictionary
        res = OrderedDict(cor=cor,tvalue=t_value,dof=dof,pvalue=p_value)
        return res
    
    #correlation matrix
    corr = cov_to_corr(X=X)
    #all combinaisons of two columns
    all_vars = combinations(X.columns,r=2)
    #correlation coefficient
    cor = DataFrame()
    for i, vars in enumerate(all_vars):
        #initialisation
        var_name = OrderedDict(Variable1=vars[0],Variable2=vars[1])
        #pearson correlation
        res = pearsonr(X=corr,var1=vars[0],var2=vars[1],n_samples=n_samples)
        #convert to DataFrame
        row_cor = DataFrame(OrderedDict(**var_name, **res),index=[i])
        #concatenate
        cor = concat((cor,row_cor),axis=0,ignore_index=True)
    #rename columns and mutate
    cor = (cor.rename(columns={"cor": "R", "tvalue": "t value", "dof" : "DF", "pvalue" : "Pr>|t|"})
              .assign(Conclusion = lambda x : where(x["Pr>|t|"] < 0.05, "Significant", "Non-significant")))
    return cor