# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryPLSLDA(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Partial Least Squares Linear Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.PLSLDA`.

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
    :class:`~discrimintools.summary.summaryPLSDA`
        Printing summaries of Partial Least Squares Discriminant Analysis model.
    :class:`~discrimintools.summary.summarySTEPDISC`
        Printing summaries of Stepwise Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_dataset
    >>> from discrimintools import PLSLDA, summaryPLSLDA
    >>> D = load_dataset("breast")
    >>> y, X = DTrain["Class"], D.drop(columns=["Class"])
    >>> clf = PLSLDA()
    >>> clf.fit(X,y)
    PLSLDA()
    >>> summaryPLSLDA(clf)
                        Partial Least Squares Linear Discriminant Analysis - Results
    Class Level Information:
              Frequency  Proportion  Prior Probability
    negative        458      0.6552             0.6552
    positive        241      0.3448             0.3448
    Importance of PLS components:
          Proportion (%)  Cumulative (%)
    Can1         69.1520         69.1520
    Can2         20.1981         89.3501
    Raw Canonical and Classification Functions Coefficients:
                 Can1    Can2  negative  positive
    Constant  -1.6329 -0.1467    1.1027   -7.2713
    ucellsize  0.2302 -0.1985   -0.4268    0.8112
    normnucl   0.2003  0.0083   -0.2664    0.5063
    mitoses    0.2119  0.4687   -0.0150    0.0285
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class PLSLDA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "plslda":
        raise ValueError("'self' must be an object of class PLSLDA")

    print("                     Partial Least Squares Linear Discriminant Analysis - Results                     ")

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
    print("\nRaw Canonical and Classification Functions Coefficients:")
    coef = concat((obj.cancoef_.raw,obj.coef_.raw),axis=1).round(decimals=digits)
    if to_markdown:
        coef = coef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(coef)

    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #manova summary
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nMultivariate Analysis of Variance (MANOVA) Summary:")
        manova = obj.lda_.statistics_.performance.round(decimals=digits)
        if to_markdown:
            manova = manova.to_markdown(tablefmt=tablefmt,**kwargs)
        print(manova)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification functions coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        class_eval = concat((obj.lda_.coef_,obj.lda_.vip_.vip),axis=1).round(decimals=digits)
        print("\nLDA Classification functions & Statistical Evaluation:")
        if to_markdown:
            class_eval = class_eval.to_markdown(tablefmt=tablefmt,**kwargs)
        print(class_eval)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)