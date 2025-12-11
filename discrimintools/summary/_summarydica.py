# -*- coding: utf-8 -*-

#intern function
from ._eval_predict import eval_predict

def summaryDiCA(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of Discriminant Correspondence Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.DiCA`.

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
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA, summaryDiCA
    >>> D = load_divay() # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    >>> summaryDiCA(clf)
                        Discriminant Correspondence Analysis - Results
    Class Level Information:
                Frequency  Proportion  Prior Probability
    Beaujolais          4      0.3333             0.3333
    Loire               4      0.3333             0.3333
    Rhone               4      0.3333             0.3333
    Importance of components:
          Eigenvalue  Difference  Proportion (%)  Cumulative (%)
    Can1      0.2519      0.0504         55.5635         55.5635
    Can2      0.2014         NaN         44.4365        100.0000
    Canonical correlation:
          Eigenvalue  Total SS  Eta Sq.  Canonical Correlation
    Can1      0.2519    4.0879   0.7394                 0.8599
    Can2      0.2014    3.4864   0.6934                 0.8327
    Classification (projection) coefficients:
                 Can1    Can2
    Woody_A   -0.3724 -0.0188
    Woody_B    0.0204  0.1559
    Woody_C    0.3520 -0.1371
    Fruity_A   0.3520 -0.1371
    Fruity_B   0.0204  0.1559
    Fruity_C  -0.3724 -0.0188
    Sweet_A    0.2017  0.2855
    Sweet_B   -0.1309 -0.0582
    Sweet_C   -0.0163 -0.1247
    Alcohol_A  0.0558 -0.3276
    Alcohol_B -0.1309 -0.0582
    Alcohol_C  0.0815  0.6236
    Hedonic_A -0.0000 -0.0000
    Hedonic_B -0.0000 -0.0000
    Hedonic_C -0.0000 -0.0000
    Hedonic_D -0.0000 -0.0000
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class DiCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "dica":
        raise ValueError("'self' must be an object of class DiCA")

    print("                     Discriminant Correspondence Analysis - Results                     ")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #class level information
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nClass Level Information:")
    class_infos = obj.classes_.infos.round(decimals=digits)
    if to_markdown:
        class_infos = class_infos.to_markdown(tablefmt=tablefmt,**kwargs)
    print(class_infos)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add eigenvalues informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nImportance of components:")
    eig = obj.eig_.iloc[:obj.call_.n_components,:].round(decimals=digits)
    if to_markdown:
        eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
    print(eig)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #canonical correlation informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nCanonical correlation:")
    cancorr = obj.cancorr_.round(decimals=digits)
    if to_markdown:
        cancorr = cancorr.to_markdown(tablefmt=tablefmt,**kwargs)
    print(cancorr)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #classification coefficients informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nClassification (projection) coefficients:")
    proj_coef = obj.cancoef_.projection.round(decimals=digits)
    if to_markdown:
        proj_coef = proj_coef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(proj_coef)

    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)