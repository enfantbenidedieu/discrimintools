# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryPLSDA(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Partial Least Squares Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.PLSDA`.

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
    :class:`~discrimintools.summary.summaryCANDISC`
        Printing summaries of Canonical Discriminant Analysis model.
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
    :class:`~discrimintools.summary.summaryPLSLDA`
        Printing summaries of Partial Least Squares Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summarySTEPDISC`
        Printing summaries of Stepwise Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_dataset
    >>> from discrimintools import PLSDA, summaryPLSDA
    >>> D = load_dataset("breast")
    >>> y, X = D["Class"], D.drop(columns=["Class"])
    >>> clf = PLSDA()
    >>> clf.fit(X,y)
    PLSDA()
    >>> summaryPLSDA(clf)
                        Partial Least Squares Discriminant Analysis - Results
    Class Level Information:
              Frequency  Proportion  Prior Probability
    negative        458      0.6552             0.6552
    positive        241      0.3448             0.3448
    Importance of PLS components:
          Proportion (%)  Cumulative (%)
    Can1         69.1520         69.1520
    Can2         20.1981         89.3501
    Classification functions coefficients:
               negative  positive     VIP
    Constant     1.0801   -0.0801     NaN
    ucellsize   -0.0853    0.0853  1.2038
    normnucl    -0.0533    0.0533  1.0346
    mitoses     -0.0030    0.0030  0.6933
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class PLSDA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "plsda":
        raise ValueError("'self' must be an object of class PLSDA")
    
    print("                     Partial Least Squares Discriminant Analysis - Results                     ")

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
    #classification fnctions coefficients
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