# -*- coding: utf-8 -*-

def summaryMPCA(
        obj,digits=4,nb_element=10,ncp=3,to_markdown=False,tablefmt = "github",**kwargs
):
    """
    Printing summaries of Mixed Principal Component Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        An object of class :class:`~discrimintools.discriminant_analysis.MPCA`.

    digits : `int <https://docs.python.org/3/library/functions.html#int>`_, default = 4
        The number of decimal printed.

    nb_element: `int <https://docs.python.org/3/library/functions.html#int>`_,, default = 10
        Number of element

    ncp : `int <https://docs.python.org/3/library/functions.html#int>`_,, default = 3 
        Number of components

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
    :class:`~discrimintools.summary.summaryGFA`
        Printing summaries of General Factor Analysis model.
    :class:`~discrimintools.summary.summaryGFALDA`
        Printing summaries of General Factor Analysis Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import MPCA, summaryMPCA
    >>> D = load_heart("subset")
    >>> X = D.drop(columns=["disease"])
    >>> clf = MPCA()
    >>> clf.fit(X)
    MPCA()
    >>> summaryMPCA(clf)
                        Mixed Principal Component Analysis - Results
    Eigenvalues informations:
                             Can1     Can2     Can3     Can4  ...     Can7     Can8     Can9     Can10
    Variance               3.5040   2.0055   1.8996   1.3246  ...   0.9318   0.8633   0.7418    0.4526
    Difference             1.4985   0.1059   0.5750   0.1522  ...   0.0684   0.1215   0.2892       NaN
    % of var.             25.0288  14.3249  13.5684   9.4616  ...   6.6555   6.1667   5.2988    3.2328
    Cumulative % of var.  25.0288  39.3536  52.9220  62.3837  ...  85.3017  91.4684  96.7672  100.0000
    [4 rows x 10 columns]
    Individuals (the 10 first):
         Can1    Can2
    0  0.4813 -0.3691
    1 -1.5953 -0.7913
    2  0.4499  5.0389
    3  1.4228  0.1799
    4  0.0194 -0.3433
    5  0.0610 -0.3194
    6 -3.8725  4.3323
    7 -2.5126 -0.3472
    8 -0.2793 -0.2635
    9 -3.4851 -1.5245
    Variables (the 10 first):
                       Can1    Can2
    age             -0.5010  0.1554
    restbpress      -0.3558  0.1148
    max_hrate        0.6681  0.0688
    asympt          -0.7169 -0.2320
    atyp_angina      0.5061  0.1254
    non_anginal      0.2706  0.0883
    typ_angina       0.1312  0.1470
    f                0.3192 -0.9171
    t               -0.3192  0.9171
    left_vent_hyper  0.0567  0.1085
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class MPCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "mpca":
        raise ValueError("'self' must be an object of class MPCA")

    print("                     Mixed Principal Component Analysis - Results                     ")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #eigenvalues informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nEigenvalues informations:")
    eig = obj.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative % of var."]
    if to_markdown:
        eig = eig.to_markdown(tablefmt=tablefmt,**kwargs)
    print(eig)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #individuals informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ind_coord = obj.ind_.coord
    if ind_coord.shape[0] > nb_element:
        print(f"\nIndividuals (the {nb_element} first):")
    else:
        print("\nIndividuals:")
    ind_coord = ind_coord.iloc[:nb_element,:min(ncp,obj.call_.n_components)].round(decimals=digits)
    if to_markdown:
        ind_coord = ind_coord.to_markdown(tablefmt=tablefmt,**kwargs)
    print(ind_coord)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #variables informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    var_coord = obj.var_.coord
    if var_coord.shape[0]>nb_element:
        print(f"\nVariables (the {nb_element} first):")
    else:
         print("\nVariables:")
    var_coord = var_coord.iloc[:nb_element,:min(ncp,obj.call_.n_components)].round(decimals=digits)
    if to_markdown:
        var_coord = var_coord.to_markdown(tablefmt=tablefmt,**kwargs)
    print(var_coord)