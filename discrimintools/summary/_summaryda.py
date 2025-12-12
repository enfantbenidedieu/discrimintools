# -*- coding: utf-8 -*-

#interns functions
from ._summarycandisc import summaryCANDISC
from ._summarycpls import summaryCPLS
from ._summarydica import summaryDiCA
from ._summarydiscrim import summaryDISCRIM
from ._summarygfalda import summaryGFALDA
from ._summarymda import summaryMDA
from ._summaryplsda import summaryPLSDA
from ._summaryplslda import summaryPLSLDA

def summaryDA(
        obj,**kwargs
):
    """
    Printing summaries of Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.CANDISC`, :class:`~discrimintools.CPLS`, :class:`~discrimintools.DiCA`, :class:`~discrimintools.DISCRIM`, :class:`~discrimintools.GFALDA`, :class:`~discrimintools.MDA`, :class:`~discrimintools.PLSDA` or :class:`~discrimintools.PLSLDA`.

    **kwargs: 
        Additionals parameters to pass to summary.

    Returns
    -------
    NoneType

    See also
    --------
    :class:`~discrimintools.summaryCANDISC`
        Printing summaries of Canonical Discriminant Analysis model.
    :class:`~discrimintools.summaryCPLS`
        Printing summaries of Partial Least Squares for Classification model.
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
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import DISCRIM, summaryDA
    >>> D = load_heart("train") # load training data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    Categorical features have been encoded into binary variables.
    DISCRIM(priors='prop')
    >>> summaryDA(clf)
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
    chestpaintypicalAngina      -2.8081   -7.4223
    restbpress                   0.3460    0.3690
    cholesteral                  0.0407    0.0373
    sugarlow                    10.4057   11.7146
    electrosttAbnormality      -15.8118  -12.8213
    electroventricHypertrophy   -1.8553   -1.2411
    maxHeartRate                 0.4856    0.4579
    ExerciseAnginayes            4.9170    5.9612
    oldpeak                      3.0652    3.7751
    slopeflat                   18.6429   19.8215
    slopeupsloping              14.7467   15.2526
    vesselsColored              -2.5087   -1.0308
    thalnormal                  21.1525   20.1211
    thalreversableEffect        14.8540   16.1636
    """
    if obj.model_ == "candisc":
        summaryCANDISC(obj=obj,**kwargs)
    elif obj.model_ == "cpls":
        summaryCPLS(obj=obj,**kwargs)
    elif obj.model_ == "dica":
        summaryDiCA(obj=obj,**kwargs)
    elif obj.model_ == "discrim":
        summaryDISCRIM(obj=obj,**kwargs)
    elif obj.model_ == "gfalda":
        summaryGFALDA(obj=obj,**kwargs)
    elif obj.model_ == "mda":
        summaryMDA(obj=obj,**kwargs)
    elif obj.model_ == "plsda":
        summaryPLSDA(obj=obj,**kwargs)
    elif obj.model_ == "plslda":
        summaryPLSLDA(obj=obj,**kwargs)
    else:
        raise ValueError("'obj' must be an object of class CANDISC, CPLS, DiCA, DISCRIM, GFALDA, MDA, PLSDA or PLSLDA")