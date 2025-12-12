# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryPLSLOGIT(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Partial Least Squares Logistic Regression model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.PLSLOGIT`.

    digits : `int <https://docs.python.org/3/library/functions.html#int>`_, default = 4
        The number of decimal printed.

    detailed : `bool <https://docs.python.org/3/library/functions.html#bool>`_, default = `False <https://docs.python.org/3/library/constants.html#False>`_
        To print detailed summaries.

    to_markdown: `bool <https://docs.python.org/3/library/functions.html#bool>`_, default = `False <https://docs.python.org/3/library/constants.html#False>`_
        To print summaries in `markdown <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html>`_-friendly format. Requires the `tabulate <https://pypi.org/project/tabulate/>`_. package.

    tablefmt : `str <https://docs.python.org/3/library/functions.html#func-str>`_, default = "github"
        The table format.

    **kwargs : 
        Additionals parameters. These parameters will be passed to `tabulate <https://pypi.org/project/tabulate/>`_.

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
    >>> from discrimintools import PLSLDA, summaryPLSLOGIT
    >>> D = load_dataset("breast")
    >>> y, X = DTrain["Class"], D.drop(columns=["Class"])
    >>> clf = PLSLOGIT()
    >>> clf.fit(X,y)
    PLSLOGIT()
    >>> summaryPLSLOGIT(clf)
    
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class PLSLOGIT
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "plslogit":
        raise ValueError("'self' must be an object of class PLSLOGIT")

    print("                     Partial Least Squares Logistic Regression - Results                     ")

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
    eig = obj.explained_variance_.round(decimals=digits)
    if to_markdown:
        eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
    print(eig)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #raw canonical coefficients and classification functions coefficients
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nRaw Canonical Coefficients:")
    cancoef = obj.cancoef_.raw.round(decimals=digits)
    if to_markdown:
        cancoef = cancoef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(cancoef)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #PLS Logistic Regression Coefficients
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nPLS Logistic Regression Coefficients:")
    coef = obj.coef_.raw.round(decimals=digits)
    #convert to DataFrame if Series
    if obj.call_.n_classes == 2:
        coef = coef.to_frame()
    if to_markdown:
        coef = coef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(coef)

    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)