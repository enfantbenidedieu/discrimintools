# -*- coding: utf-8 -*-

#intern function
from ._summaryda import summaryDA

def summarySTEPDISC(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt = "github",**kwargs
):
    """
    Printing summaries of Stepwise Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.STEPDISC`.

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
    :class:`~discrimintools.summary.summaryPLSDA`
        Printing summaries of Partial Least Squares Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryPLSLDA`
        Printing summaries of Partial Least Squares Linear Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_dataset
    >>> from discrimintools import DISCRIM, STEPDISC, summarySTEPDISC
    >>> D = load_dataset("breast")
    >>> y, X = D["Class"], D.drop(columns=["Class"])
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    DISCRIM(priors='prop')
    >>> clf2 = STEPDISC(method="backward",alpha=0.01,verbose=False)
    >>> clf2.fit(clf)
    STEPDISC()
    >>> summarySTEPDISC(clf2)
                        Stepwise Discriminant Analysis - Results
    ====================== Before backward selection  =======================
                        Discriminant Analysis - Results
    Summary Information:
                   Infos  Value                  DF  DF value
    0  Total Sample Size    150            DF Total       149
    1          Variables     18   DF Within Classes       148
    2            Classes      2  DF Between Classes         1
    Class Level Information:
              Frequency  Proportion  Prior Probability
    absence          82      0.5467             0.5467
    presence         68      0.4533             0.4533
    Linear Discriminant Function for disease:
                                absence  presence
    Constant                  -124.3546 -127.3863
    age                          1.1836    1.1914
    sexmale                     14.2659   15.9123
    chestpainatypicalAngina      0.5668   -1.8839
    chestpainnonAnginal          3.4872    1.6652
    ...
    slopeflat                   18.6429   19.8215
    slopeupsloping              14.7467   15.2526
    vesselsColored              -2.5087   -1.0308
    thalnormal                  21.1525   20.1211
    thalreversableEffect        14.8540   16.1636
    ====================== After backward selection  =======================
                        Discriminant Analysis - Results
    Summary Information:
                   Infos  Value                  DF  DF value
    0  Total Sample Size    150            DF Total       149
    1          Variables      7   DF Within Classes       148
    2            Classes      2  DF Between Classes         1
    Class Level Information:
              Frequency  Proportion  Prior Probability
    absence          82      0.5467             0.5467
    presence         68      0.4533             0.4533
    Linear Discriminant Function for disease:
                             absence  presence
    Constant                 -3.5002   -6.2648
    sexmale                   3.0078    4.6211
    chestpainatypicalAngina   4.6553    1.9352
    chestpainnonAnginal       4.6453    2.1583
    chestpaintypicalAngina    2.6881   -2.0794
    oldpeak                   0.7868    1.8112
    vesselsColored            0.7138    2.0291
    thalreversableEffect      0.6085    2.4439
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class STEPDISC
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "stepdisc":
        raise ValueError("'self' must be an object of class STEPDISC")

    print("                     Stepwise Discriminant Analysis - Results                     ")

    print("\n====================== Before {} selection  =======================\n".format(obj.method))
    summaryDA(obj.call_.obj,digits=digits,to_markdown=to_markdown,tablefmt=tablefmt,detailed=detailed,**kwargs)

    if hasattr(obj,"disc_"):
        print("\n====================== After {} selection  =======================\n".format(obj.method))
        summaryDA(obj.disc_,digits=digits,to_markdown=to_markdown,tablefmt=tablefmt,detailed=detailed,**kwargs)
    else:
        print("\nNo model has been updated.")