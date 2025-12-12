# -*- coding: utf-8 -*-

#intern function
from ._eval_predict import eval_predict

def summaryDISCRIM(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Discriminant Analysis (linear and quadratic) model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        an object of class :class:`~discrimintools.DISCRIM`.

    digits : `int <https://docs.python.org/3/library/functions.html#int>`_, default = 4
        The number of decimal printed.

    detailed : `bool <https://docs.python.org/3/library/functions.html#bool>`_, default = `False <https://docs.python.org/3/library/constants.html#False>`_
        To print detailed summaries.

    to_markdown: `bool <https://docs.python.org/3/library/functions.html#bool>`_, default = `False <https://docs.python.org/3/library/constants.html#False>`_
        To print summaries in `markdown <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html>`_-friendly format. Requires the `tabulate <https://pypi.org/project/tabulate/>`_. package.

    tablefmt : `str <https://docs.python.org/3/library/functions.html#func-str>`_, default = "github"
        The table format.

    **kwargs : 
        additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

    Returns
    -------
    NoneType

    See also
    --------
    :class:`~discrimintools.summaryCANDISC`
        Printing summaries of Canonical Discriminant Analysis model.
    :class:`~discrimintools.summaryCPLS`
        Printing summaries of Partial Least Squares for Classification model.
    :class:`~discrimintools.summaryDA`
        Printing summaries of Discriminant Analysis model.
    :class:`~discrimintools.summaryDiCA`
        Printing summaries of Discriminant Correspondence Analysis model.
    :class:`~discrimintools.summaryGFALDA`
        Printing summaries of General Factor Analysis Linear Discriminant Analysis model.
    :class:`~discrimintools.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.
    :class:`~discrimintools.summaryPLSDA`
        Printing summaries of Partial Least Squares Discriminant Analysis model.
    :class:`~discrimintools.summaryPLSLDA`
        Printing summaries of Partial Least Squares Linear Discriminant Analysis model.
    :class:`~discrimintools.summarySTEPDISC`
        Printing summaries of Stepwise Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_alcools
    >>> from discrimintools import DISCRIM, summaryDISCRIM
    >>> D = load_alcools()
    >>> y, X = D["TYPE"], D.drop(columns=["TYPE"])
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    DISCRIM(priors='prop')
    >>> summaryDISCRIM(clf)
                        Discriminant Analysis - Results
    Summary Information:
                   Infos  Value                  DF  DF value
    0  Total Sample Size     52            DF Total        51
    1          Variables      8   DF Within Classes        49
    2            Classes      3  DF Between Classes         2
    Class Level Information:
            Frequency  Proportion  Prior Probability
    KIRSCH         17      0.3269             0.3269
    MIRAB          15      0.2885             0.2885
    POIRE          20      0.3846             0.3846
    Linear Discriminant Function for TYPE:
              KIRSCH    MIRAB    POIRE
    Constant -5.0165 -18.8407 -24.7649
    MEOH      0.0034   0.0290   0.0334
    ACET      0.0064   0.0164   0.0075
    BU1      -0.0637   0.4054   0.3180
    BU2      -0.0009   0.0714   0.1150
    ISOP      0.0231   0.0298  -0.0085
    MEPR      0.0375  -0.1289   0.0618
    PRO1      0.0020  -0.0054  -0.0083
    ACAL      0.0662  -0.2264  -0.1303
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an object of class DISCRIM
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "discrim":
        raise ValueError("'obj' must be an object of class DISCRIM")

    print("                     Discriminant Analysis - Results                     ")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #summary information
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nSummary Information:")
    summary = obj.summary_.infos
    if to_markdown:
        summary = summary.to_markdown(tablefmt=tablefmt,**kwargs) 
    print(summary)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #class level information
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nClass Level Information:")
    class_infos = obj.classes_.infos.round(decimals=digits)
    if to_markdown:
        class_infos = class_infos.to_markdown(tablefmt=tablefmt,**kwargs)
    print(class_infos)
    
    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #covariance matrix information
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.method == "linear":
            print("\nPooled Covariance Matrix Information:")
            cov_mat = obj.cov_.infos.loc["Pooled",:].to_frame().T.round(decimals=digits)
        else:
            print("\nWithin Covariance Matrix Information:")
            cov_mat = obj.cov_.infos.round(decimals=digits)
        cov_mat["Rank"] = cov_mat["Rank"].astype(int)
        if to_markdown:
            cov_mat = cov_mat.to_markdown(tablefmt=tablefmt,**kwargs)
        print(cov_mat)

        if obj.method == "quad":
            print("\nTest of Homogeneity of Within Covariance Matrices:")
            equal_cov = obj.cov_.test.round(decimals=digits)
            if to_markdown:
                equal_cov = equal_cov.to_markdown(tablefmt=tablefmt,**kwargs)
            print(equal_cov)

            #chi-square pvalue
            p_value, tol = obj.cov_.test.iloc[0,6], obj.call_.tol
            if p_value > tol:
                print(f"\nSince the Chi-Square value is not significant at the {tol} level, a pooled covariance matrix has been used in the discriminant function.\nReference: Morrison, D.F. (1976) Multivariate Statistical Methods p252.")
            elif p_value <= tol:
                print(f"\nSince the Chi-Square value is significant at the {tol} level, the within covariance matrices has been used in the discriminant function.\nReference: Morrison, D.F. (1976) Multivariate Statistical Methods p252.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #linear discriminant function
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p_value, tol = obj.cov_.test.iloc[0,6], obj.call_.tol
    if (obj.method == "linear") or (obj.method == "quad" and  p_value > tol):
        print("\nLinear Discriminant Function for {}:".format(obj.call_.target))
        coef = obj.coef_.round(decimals=digits)
        if to_markdown:
            coef = coef.to_markdown(tablefmt=tablefmt,**kwargs)
        print(coef)

    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)