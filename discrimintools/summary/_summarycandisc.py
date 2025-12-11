# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryCANDISC(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Canonical Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.CANDISC`.

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
    :class:`~discrimintools.summary.summaryCPLS`
        Printing summaries of Partial Least Squares for Classification model.
    :class:`~discrimintools.summary.summaryDA`
        Printing summaries of Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryDiCA`
        Printing summaries of Discriminant Correspondence Analysis model.
    :class:`~discrimintools.summary.summaryDISCRIM`
        Printing summaries of Discriminant Analysis (linear and quadratic) model.
    :class:`~discrimintools.summary.summaryGFALDA`
        Printing summaries of General Factor Analysis Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryPLSDA`
        Printing summaries of Partial Least Squares Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryPLSLDA`
        Printing summaries of Partial Least Squares Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summarySTEPDISC`
        Printing summaries of Stepwise Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, summaryCANDISC
    >>> D = load_wine() # load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> summaryCANDISC(clf)
                            Canonical Discriminant Analysis - Results
    Summary Information:
                   infos  Value                  DF  DF value
    0  Total Sample Size     34            DF Total        33
    1          Variables      4   DF Within Classes        31
    2            Classes      3  DF Between Classes         2
    Class Level Information:
              Frequency  Proportion  Prior Probability
    Mediocre         12      0.3529             0.3529
    Moyen            11      0.3235             0.3235
    Bon              11      0.3235             0.3235
    Total-Sample Class Means:
                  Mediocre      Moyen        Bon
    Temperature  3037.3333  3140.9091  3306.3636
    Soleil       1126.4167  1262.9091  1363.6364
    Chaleur        12.0833    16.4545    28.5455
    Pluie         430.3333   339.6364   305.0000
    Importance of components:
          Eigenvalue  Difference  Proportion  Cumulative
    Can1      3.2789      3.1403     95.9451     95.9451
    Can2      0.1386         NaN      4.0549    100.0000
    Raw Canonical and Classification Functions Coefficients:
                    Can1    Can2  Mediocre   Moyen      Bon
    Constant    -32.8763  2.1653   65.6093 -7.1918 -72.5905
    Temperature   0.0086 -0.0000   -0.0178  0.0013   0.0182
    Soleil        0.0068 -0.0053   -0.0153  0.0037   0.0129
    Chaleur      -0.0271  0.1276    0.0845 -0.0694  -0.0227
    Pluie        -0.0059  0.0062    0.0136 -0.0040  -0.0108
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class CANDISC
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "candisc":
        raise ValueError("'self' must be an object of class CANDISC")

    print("                     Canonical Discriminant Analysis - Results                     ")

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

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Class Means
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nTotal-Sample Class Means:")
    wcenter = obj.classes_.center.T.round(decimals=digits)
    if to_markdown:
        wcenter = wcenter.to_markdown(tablefmt=tablefmt,**kwargs)
    print(wcenter)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add eigenvalues informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nImportance of components:")
    eig = obj.eig_.iloc[:obj.call_.n_components,:].round(decimals=digits)
    if to_markdown:
        eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
    print(eig)

    print("\nRaw Canonical and Classification Functions Coefficients:")
    coef = concat((obj.cancoef_.raw,obj.coef_),axis=1).round(decimals=digits)
    if to_markdown:
        coef = coef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(coef)

    if detailed:
        print("\nTest of H0: The canonical correlations in the current row and all that follow are zero")
        cancorr = obj.cancorr_.round(decimals=digits)
        if to_markdown:
            cancorr = cancorr.to_markdown(tablefmt=tablefmt,**kwargs)
        print(cancorr)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)