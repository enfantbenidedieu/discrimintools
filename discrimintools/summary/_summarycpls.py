# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryCPLS(
        obj,ncp=2,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Partial Least Squares for Classification model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.CPLS`.

    ncp : `int <https://docs.python.org/3/library/functions.html#int>`_,, default = 2
        Number of pls components.

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
    :class:`~discrimintools.summaryDA`
        Printing summaries of Discriminant Analysis model.
    :class:`~discrimintools.summaryDiCA`
        Printing summaries of Discriminant Correspondence Analysis model.
    :class:`~discrimintools.summaryDISCRIM`
        Printing summaries of Discriminant Analysis (linear and quadratic) model.
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
    >>> from discrimintools.datasets import load_dataset
    >>> from discrimintools import CPLS, summaryCPLS
    >>> DTrain = load_dataset("breast") # load training data
    >>> y, X = D["Class"], D.drop(columns=["Class"]) # split into X and y
    >>> clf = CPLS()
    >>> clf.fit(X,y)
    CPLS()
    >>> summaryCPLS(clf)
                            Partial Least Squares for Classification - Results
    Class Level Information:
              Frequency  Proportion  Prior Probability
    negative        458      0.6552             0.6552
    positive        241      0.3448             0.3448
    Classification functions coefficients:
               positive       VIP
    Constant  -0.424881       NaN
    ucellsize  0.085323  1.203759
    normnucl   0.053251  1.034559
    mitoses    0.003001  0.693292
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class CPLS
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "cpls":
        raise ValueError("'self' must be an object of class CPLS")

    print("                     Partial Least Squares for Classification - Results                     ")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #class level information
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nClass Level Information:")
    class_infos = obj.classes_.infos.round(decimals=digits)
    if to_markdown:
        class_infos = class_infos.to_markdown(tablefmt=tablefmt,**kwargs)
    print(class_infos)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #importance of pls components
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nImportance of PLS components:")
    eig = obj.explained_variance_.iloc[:min(ncp,obj.call_.n_components),:].round(decimals=digits)
    if to_markdown:
        eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
    print(eig)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #classification functions coefficients
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nClassification functions coefficients:")
    coef = concat((obj.coef_, obj.vip_.vip),axis=1).round(decimals=digits)
    if to_markdown:
        coef = coef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(coef)

    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)