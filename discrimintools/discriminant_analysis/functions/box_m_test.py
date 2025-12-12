# -*- coding: utf-8 -*-
from pandas import DataFrame
from statsmodels.stats.multivariate import test_cov_oneway

def box_m_test(W,n_k):
    """
    Box-M test

    Performs multiple sample hypothesis test that covariance matrices are equal. The Null and alternative hypotheses are

    .. math::
       H0 &: \Sigma_i = \Sigma_j  \text{ for all i and j} \\
       H1 &: \Sigma_i \neq \Sigma_j \text{ for at least one i and j}

    where :math:`\Sigma_i` is the covariance of sample `i`.

    Parameters
    ----------
    W : list
        Within-class covariance matrices, estimated with denominator `N-1`, i.e. `ddof=1`.

    n_k : list
        List of the number of observations used in the estimation of the covariance for each sample.

    Returns
    -------
    res : DataFrame
        Box's M tests for homogeneity of variance-covariance matrices

    References
    ----------
    [1] Rencher, Alvin C., and William F. Christensen. 2012. Methods of Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc. https://doi.org/10.1002/9781118391686.

    [2] StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.

    [3] Stata Press Publication.
    ```
    """
    #box's m test
    box_m = test_cov_oneway(W,n_k)
    ddl1, ddl2 = (int(x) for x in box_m.df_f)
    columns = ["Bartlett Value","Num DF","Den DF","F value", "Pr>F", "Chi Sq. Value", "Pr>Chi2"]
    res = DataFrame([[box_m.statistic_base,ddl1,ddl2,box_m.statistic_f, box_m.pvalue_f,box_m.statistic_chi2, box_m.pvalue_chi2]],columns=columns,index=["Box's M"])
    return res