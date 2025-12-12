# -*- coding: utf-8 -*-
from pandas import concat

#intern function
from ._eval_predict import eval_predict

def summaryGFALDA(
        obj,digits=4,detailed=False,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of General Factor Analysis Linear Discriminant Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        an object of class :class:`~discrimintools.GFALDA`.

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
    :class:`~discrimintools.summaryCPLS`
        Printing summaries of Partial Least Squares for Classification model.
    :class:`~discrimintools.summaryDA`
        Printing summaries of Discriminant Analysis model.
    :class:`~discrimintools.summaryDiCA`
        Printing summaries of Discriminant Correspondence Analysis model.
    :class:`~discrimintools.summaryDISCRIM`
        Printing summaries of Discriminant Analysis (linear and quadratic) model.
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
    >>> from discrimintools.datasets import load_alcools, load_vote, load_heart
    >>> from discrimintools import GFALDA, summaryGFALDA

    The :class:`~discrimintools.GFALDA` class performs principal component analysis - discriminant analysis (PCADA)...

    >>> #PCA + LDA = PCADA
    >>> D = load_alcools("train")
    >>> y, X = D["TYPE"], D.drop(columns=["TYPE"])
    >>> clf = GFALDA()
    >>> clf.fit(X,y)
    GFALDA()
    >>> summaryGFALDA(clf)
                        General Factor Analysis Linear Discriminant Analysis - Results
    Class Level Information:
            Frequency  Proportion  Prior Probability
    KIRSCH         17      0.3269             0.3269
    MIRAB          15      0.2885             0.2885
    POIRE          20      0.3846             0.3846
    Importance of components:
          Eigenvalue  Difference  Proportion (%)  Cumulative (%)
    Can1      2.7988      1.0799         34.9848         34.9848
    Can2      1.7188      0.3154         21.4856         56.4703
    Raw Canonical Coefficients:
                Can1    Can2
    Constant -3.9272 -2.1923
    MEOH      0.0014  0.0001
    ACET      0.0005  0.0029
    BU1       0.0418 -0.0025
    BU2      -0.0005  0.0107
    ISOP      0.0096  0.0002
    MEPR      0.0264  0.0048
    PRO1     -0.0003  0.0010
    ACAL      0.0230  0.0432
    Projection functions coefficients:
            Can1    Can2
    MEOH  0.0651  0.0057
    ACET  0.0076  0.0434
    BU1   0.0570 -0.0034
    BU2  -0.0037  0.0721
    ISOP  0.0575  0.0012
    MEPR  0.0604  0.0110
    PRO1 -0.0244  0.0807
    ACAL  0.0230  0.0431

    discriminant analysis on qualitative variables (`DISQUAL <https://lemakistatheux.wordpress.com/2016/12/07/lalgorithme-disqual/>`_)...

    >>> #MCA + LDA = DISQUAL
    >>> D = load_vote("train")
    >>> y, X = D["group"], D.drop(columns=["group"])
    >>> clf = GFALDA(n_components=5)
    >>> clf.fit(X,y)
    GFALDA(n_components=5)
    >>> summaryGFALDA(clf)
                        General Factor Analysis Linear Discriminant Analysis - Results
    Class Level Information:
                Frequency  Proportion  Prior Probability
    democrat          154      0.6553             0.6553
    republican         81      0.3447             0.3447
    Importance of components:
          Eigenvalue  Difference  Proportion (%)  Cumulative (%)
    Can1      0.4912      0.2018         24.5584         24.5584
    Can2      0.2893      0.1926         14.4672         39.0255
    Can3      0.0968      0.0023          4.8380         43.8635
    Can4      0.0945      0.0095          4.7226         48.5861
    Can5      0.0850      0.0065          4.2499         52.8360
    Raw Canonical Coefficients:
                                        Can1      Can2      Can3      Can4      Can5
    Constant                         -3.6423  -71.5330   -0.5466   -0.3136   -8.7288
    handicapped_infants_n             1.4104   -0.0444   -1.3628   -1.3368   -0.9953
    handicapped_infants_other        -7.7080  112.3077  228.0942 -137.3374  118.7663
    handicapped_infants_y            -1.8593   -0.5920    0.5157    2.6020    0.6542
    water_project_cost_sharin_n      -0.3863   -0.8587   -3.4385   -3.4936    0.8114
    ...
    duty_free_exports_other           6.4025   76.2179  -70.0049  -26.0326  -51.8400
    duty_free_exports_y              -2.6233   -0.4362   -0.3366   -0.4600   -0.1570
    export_administration_act_n      14.1955   -2.4783    9.4113   10.6011   -8.8429
    export_administration_act_other  -2.4658    3.0700   -2.1882    6.4082    2.9454
    export_administration_act_y      -0.1613   -0.4680   -0.0029   -1.6712   -0.1645
    Projection functions coefficients:
                                       Can1    Can2    Can3    Can4    Can5
    handicapped_infants_n            0.0458 -0.0014 -0.0442 -0.0434 -0.0323
    handicapped_infants_other       -0.0164  0.2390  0.4853 -0.2922  0.2527
    handicapped_infants_y           -0.0519 -0.0165  0.0144  0.0727  0.0183
    water_project_cost_sharin_n     -0.0106 -0.0235 -0.0942 -0.0957  0.0222
    water_project_cost_sharin_other -0.0006  0.1209  0.1482 -0.1715  0.0665
    ...
    duty_free_exports_other          0.0204  0.2432 -0.2234 -0.0831 -0.1654
    duty_free_exports_y             -0.0712 -0.0118 -0.0091 -0.0125 -0.0043
    export_administration_act_n      0.1133 -0.0198  0.0751  0.0846 -0.0706
    export_administration_act_other -0.0407  0.0506 -0.0361  0.1057  0.0486
    export_administration_act_y     -0.0061 -0.0178 -0.0001 -0.0636 -0.0063

    and discriminant analysis on mixed predictors (DISMIX).

    >>> #FAMD + LDA = DISMIX
    >>> D = load_heart("subset")
    >>> y, X = D["disease"], D.drop(columns=["disease"])
    >>> clf = GFALDA(n_components=5)
    >>> clf.fit(X,y)
    GFALDA(n_components=5)
    >>> summaryGFALDA(clf)
                        General Factor Analysis Linear Discriminant Analysis - Results
    Class Level Information:
              Frequency  Proportion  Prior Probability
    negative        117      0.5598             0.5598
    positive         92      0.4402             0.4402
    Importance of components:
          Eigenvalue  Difference  Proportion (%)  Cumulative (%)
    Can1      2.3749      1.2131         23.7495         23.7495
    Can2      1.1618      0.0978         11.6184         35.3678
    Can3      1.0641      0.0398         10.6408         46.0086
    Can4      1.0243      0.0441         10.2431         56.2517
    Can5      0.9802      0.0131          9.8024         66.0541
    Raw Canonical Coefficients:
                             Can1    Can2    Can3    Can4    Can5
    Constant              -1.8294 -4.1264 -2.2226 -3.4631 -3.2657
    age                    0.0518  0.0394 -0.0015  0.0005  0.0217
    restbpress             0.0170  0.0138  0.0133  0.0246  0.0163
    max_hrate             -0.0213  0.0028  0.0038  0.0011  0.0003
    asympt                 0.4341 -0.3523 -0.1561 -0.3150 -0.2125
    ...
    left_vent_hyper       -0.5277  3.1395 -2.1900  0.6887 -3.1521
    normal                -0.0688 -0.0481 -0.0640 -0.2254  0.0214
    st_t_wave_abnormality  0.4870 -0.2444  0.7363  1.1923  0.4011
    no                    -0.3450  0.1151  0.0791  0.0815  0.1269
    yes                    0.6565 -0.2189 -0.1505 -0.1550 -0.2415
    Projection functions coefficients:
                             Can1    Can2    Can3    Can4    Can5
    age                    0.1389  0.1054 -0.0040  0.0012  0.0581
    restbpress             0.0983  0.0803  0.0771  0.1428  0.0944
    max_hrate             -0.1687  0.0224  0.0298  0.0086  0.0028
    asympt                 0.1672 -0.0949 -0.0403 -0.0797 -0.0526
    atyp_angina           -0.1671 -0.0151  0.1161  0.2078 -0.1317
    ...
    left_vent_hyper       -0.2033  0.8460 -0.5648  0.1743 -0.7802
    normal                -0.0265 -0.0130 -0.0165 -0.0570  0.0053
    st_t_wave_abnormality  0.1876 -0.0659  0.1899  0.3017  0.0993
    no                    -0.1329  0.0310  0.0204  0.0206  0.0314
    yes                    0.2529 -0.0590 -0.0388 -0.0392 -0.0598
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class GFALDA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "gfalda":
        raise ValueError("'self' must be an object of class GFALDA")

    print("                     General Factor Analysis Linear Discriminant Analysis - Results                     ")

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
    eig = obj.pipe_["gfa"].eig_.iloc[:obj.call_.n_components,:].round(decimals=digits)
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
        print("\nLDA Classification functions & Statistical Evaluation:")
        class_eval = concat((obj.pipe_["lda"].coef_,obj.pipe_["lda"].vip_.vip),axis=1).round(decimals=digits)
        if to_markdown:
            class_eval = class_eval.to_markdown(tablefmt=tablefmt,**kwargs)
        print(class_eval)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification summary for calibration data
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eval_predict(obj,digits,to_markdown,tablefmt,**kwargs)