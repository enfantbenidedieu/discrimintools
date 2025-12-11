# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryMDA(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """Printing summaries of Mixed Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.MDA`.

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
    :class:`~discrimintools.summary.summaryPLSDA`
        Printing summaries of Partial Least Squares Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryPLSLDA`
        Printing summaries of Partial Least Squares Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summarySTEPDISC`
        Printing summaries of Stepwise Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import MDA, summaryMDA
    >>> D = load_heart("subset") # load training data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = MDA(n_components=5)
    >>> clf.fit(X,y)
    MDA(n_components=5)
    >>> summaryMDA(clf)
                        Mixed Discriminant Analysis - Results
    Importance of components:
          Eigenvalue  Difference  Proportion (%)  Cumulative (%)
    Can1      3.5040      1.4985         25.0288         25.0288
    Can2      2.0055      0.1059         14.3249         39.3536
    Can3      1.8996      0.5750         13.5684         52.9220
    Can4      1.3246      0.1522          9.4616         62.3837
    Can5      1.1724      0.0681          8.3743         70.7580
    Raw Canonical Coefficients:
                             Can1    Can2    Can3    Can4    Can5
    Constant              -0.0059  0.2104  0.2151 -0.8402 -5.4678
    age                   -0.0333  0.0137  0.0030  0.0189  0.0549
    restbpress            -0.0109  0.0047  0.0056 -0.0012  0.0297
    max_hrate              0.0150  0.0020  0.0027  0.0007 -0.0121
    asympt                -0.7662 -0.3278 -0.4341 -0.0563 -0.5999
    ...
    left_vent_hyper        0.1982  0.5016  1.7704  1.4714 -1.7099
    normal                 0.4200  0.4352 -1.7807 -0.1179  0.2904
    st_t_wave_abnormality -0.5140 -0.5888  1.6832 -0.1457 -0.0045
    no                     0.9448  0.1881  0.2183  0.1125  0.1830
    yes                   -0.9448 -0.1881 -0.2183 -0.1125 -0.1830
    Projection functions coefficients:
                             Can1    Can2    Can3    Can4    Can5
    age                   -0.0892  0.0366  0.0081  0.0507  0.1472
    restbpress            -0.0634  0.0270  0.0322 -0.0067  0.1723
    max_hrate              0.1190  0.0162  0.0218  0.0054 -0.0965
    asympt                -0.0957 -0.0410 -0.0542 -0.0070 -0.0750
    atyp_angina            0.0676  0.0221  0.0450 -0.1487  0.0679
    ...
    left_vent_hyper        0.0076  0.0192  0.0676  0.0562 -0.0653
    normal                 0.0392  0.0406 -0.1662 -0.0110  0.0271
    st_t_wave_abnormality -0.0451 -0.0516  0.1475 -0.0128 -0.0004
    no                     0.1122  0.0224  0.0259  0.0134  0.0217
    yes                   -0.1122 -0.0224 -0.0259 -0.0134 -0.0217
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class MDA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "mda":
        raise ValueError("'self' must be an object of class MDA")

    print("                     Mixed Discriminant Analysis - Results                     ")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add eigenvalues informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nImportance of components:")
    eig = obj.pipe_["mpca"].eig_.iloc[:obj.call_.n_components,:].round(decimals=digits)
    if to_markdown:
        eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
    print(eig)

    print("\nRaw Canonical Coefficients:")
    can_coef_raw = obj.cancoef_.raw.round(decimals=digits)
    if to_markdown:
        can_coef_raw = can_coef_raw.to_markdown(tablefmt=tablefmt,**kwargs)
    print(can_coef_raw)
        
    print("\nProjection functions coefficients:")
    proj_coef = obj.cancoef_.projection.round(decimals=digits)
    if to_markdown:
        proj_coef = proj_coef.to_markdown(tablefmt=tablefmt,**kwargs)
    print(proj_coef)

    if detailed:
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #manova summary
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print("\nMultivariate Analysis of Variance (MANOVA) Summary:")
        manova = obj.pipe_["lda"].statistics_.performance.round(decimals=digits)
        if to_markdown:
            manova = manova.to_markdown(tablefmt=tablefmt,**kwargs)
        print(manova)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification functions coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        class_eval = concat((obj.pipe_["lda"].coef_,obj.pipe_["lda"].vip_.vip),axis=1).round(decimals=digits)
        print("\nLDA Classification functions & Statistical Evaluation:")
        if to_markdown:
            class_eval = class_eval.to_markdown(tablefmt=tablefmt,**kwargs)
        print(class_eval)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)