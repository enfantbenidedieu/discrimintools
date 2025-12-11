# -*- coding: utf-8 -*-
def eval_predict(
        obj,digits=4,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Evaluation of the prediction' quality on training dataset.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        an object of class :class:`~discrimintools.discriminant_analysis.CANDISC`, :class:`~discrimintools.discriminant_analysis.CPLS`, :class:`~discrimintools.discriminant_analysis.DiCA`, :class:`~discrimintools.discriminant_analysis.DISCRIM`, :class:`~discrimintools.discriminant_analysis.GFALDA`, :class:`~discrimintools.discriminant_analysis.MDA`, :class:`~discrimintools.discriminant_analysis.PLSDA` or :class:`~discrimintools.discriminant_analysis.PLSLDA`.

    digits : `int <https://docs.python.org/3/library/functions.html#int>`_, default = 4
        The number of decimal printed.

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
    :class:`~discrimintools.summary.summaryPLSDA`
        Printing summaries of Partial Least Squares Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryPLSLDA`
        Printing summaries of Partial Least Squares Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summarySTEPDISC`
        Printing summaries of Stepwise Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import DISCRIM, eval_predict
    >>> D = load_heart() # load training data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    Categorical features have been encoded into binary variables.
    DISCRIM(priors='prop')
    >>> eval_predict(clf)
    Classification Summary for Calibration Data:
    Observation Profile:
                            Read  Used
    Number of Observations   150   150
    Number of Observations Classified into disease:
    prediction  absence  presence  Total
    disease
    absence          75         7     82
    presence         12        56     68
    Total            87        63    150
    Percent Classified into disease:
    prediction  absence  presence  Total
    disease
    absence     91.4634    8.5366  100.0
    presence    17.6471   82.3529  100.0
    Total       58.0000   42.0000  100.0
    Priors       0.5467    0.4533    NaN
    Error Count Estimates for disease:
            absence  presence   Total
    Rate     0.0854    0.1765  0.1267
    Priors   0.5467    0.4533     NaN
    Classification Report for disease:
                  precision  recall  f1-score   support
    absence          0.8621  0.9146    0.8876   82.0000
    presence         0.8889  0.8235    0.8550   68.0000
    accuracy         0.8733  0.8733    0.8733    0.8733
    macro avg        0.8755  0.8691    0.8713  150.0000
    weighted avg     0.8742  0.8733    0.8728  150.0000
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #classification summary for calibration data
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nClassification Summary for Calibration Data:\n")
    evl = obj.eval_predict(obj.call_.Xtot,obj.call_.y,verbose=False)
    print("Observation Profile:")
    obs = evl.obs
    if to_markdown:
        obs = obs.to_markdown(tablefmt=tablefmt,**kwargs)
    print(obs)

    print("\nNumber of Observations Classified into {}:".format(obj.call_.target))
    cm = evl.cm
    if to_markdown:
        cm = cm.to_markdown(tablefmt=tablefmt,**kwargs)
    print(cm)
    
    print("\nPercent Classified into {}:".format(obj.call_.target))
    resub = evl.resub.round(decimals=digits)
    if to_markdown:
        resub = resub.to_markdown(tablefmt=tablefmt,**kwargs)
    print(resub)

    print("\nError Count Estimates for {}:".format(obj.call_.target))
    error = evl.error.round(decimals=digits)
    if to_markdown:
        error = error.to_markdown(tablefmt=tablefmt,**kwargs)
    print(error)

    print("\nClassification Report for {}:".format(obj.call_.target))
    report = evl.report.round(decimals=digits)
    if to_markdown:
        report = report.to_markdown(tablefmt=tablefmt,**kwargs) 
    print(report)