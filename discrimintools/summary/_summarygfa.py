# -*- coding: utf-8 -*-

def summaryGFA(
        obj,digits=4,nb_element=10,ncp=3,to_markdown=False,tablefmt="github",**kwargs
):
    """
    Printing summaries of General Factor Analysis model.

    Parameters
    ----------
    obj : `class <https://docs.python.org/3/tutorial/classes.html>`_
        an object of class :class:`~discrimintools.discriminant_analysis.GFA`.

    digits : `int <https://docs.python.org/3/library/functions.html#int>`_, default = 4
        The number of decimal printed.

    nb_element: `int <https://docs.python.org/3/library/functions.html#int>`_,, default = 10. 
        Number of element.

    ncp : `int <https://docs.python.org/3/library/functions.html#int>`_,, default = 3. 
        Number of components.

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
    :class:`~discrimintools.summary.summaryGFALDA`
        Printing summaries of General Factor Analysis Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryMPCA`
        Printing summaries of Mixed Principal Component Analysis model.
    :class:`~discrimintools.summary.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.

    Examples
    --------
    >>> from discrimintools.datasets import load_alcools, load_canines, load_heart
    >>> from discrimintools import GFA, summaryGFA

    The :class:`~discrimintools.discriminant_analysis.GFA` performs principal component analysis (PCA) ...

    >>> #PCA
    >>> D = load_alcools("train")
    >>> X = D.drop(columns=["TYPE"])
    >>> clf = GFA()
    >>> clf.fit(X) 
    GFA()
    >>> summaryGFA(clf)
                        General Factor Analysis - Results
    Eigenvalues informations:
                             Can1     Can2     Can3     Can4     Can5     Can6     Can7      Can8
    Variance               2.7988   1.7188   1.4034   1.0444   0.5368   0.2024   0.1851    0.1103
    Difference             1.0799   0.3154   0.3590   0.5076   0.3344   0.0173   0.0748       NaN
    % of var.             34.9848  21.4856  17.5428  13.0554   6.7101   2.5295   2.3136    1.3782
    Cumulative % of var.  34.9848  56.4703  74.0132  87.0686  93.7786  96.3081  98.6218  100.0000
    Individuals (the 10 first):
         Can1    Can2
    0 -1.4901 -1.1150
    1 -0.8484  1.0139
    2 -1.7262  0.6568
    3 -1.7259 -1.1717
    4 -3.6258 -1.4189
    5 -0.9469  2.3607
    6 -0.7407  1.8183
    7 -3.4476 -1.4497
    8 -1.6847 -0.9451
    9 -3.7593 -1.0701
    Variables:
            Can1    Can2
    MEOH  0.8711  0.0601
    ACET  0.1017  0.4556
    BU1   0.7630 -0.0353
    BU2  -0.0493  0.7563
    ISOP  0.7692  0.0130
    MEPR  0.8078  0.1149
    PRO1 -0.3263  0.8465
    ACAL  0.3073  0.4522

    multiple correspondence analysis (MCA)...

    >>> #MCA
    >>> D = load_canines("train")
    >>> X = D.drop(columns=["Fonction"])
    >>> clf = GFA()
    >>> clf.fit(X)
    GFA()
    >>> summaryGFA(clf)
                        General Factor Analysis - Results
    Eigenvalues informations:
                             Can1     Can2     Can3     Can4  ...     Can7     Can8     Can9     Can10
    Variance               0.4816   0.3847   0.2110   0.1576  ...   0.0815   0.0457   0.0235    0.0077
    Difference             0.0969   0.1738   0.0534   0.0074  ...   0.0358   0.0221   0.0158       NaN
    % of var.             28.8964  23.0842  12.6572   9.4532  ...   4.8877   2.7402   1.4125    0.4628
    Cumulative % of var.  28.8964  51.9806  64.6379  74.0911  ...  95.3845  98.1247  99.5372  100.0000
    [4 rows x 10 columns]
    Individuals (the 10 first):
                   Can1    Can2
    Beauceron   -0.3172 -0.4177
    Basset       0.2541  1.1012
    Berger All  -0.4864 -0.4644
    Boxer        0.4474 -0.8818
    Bull-Dog     1.0134  0.5499
    Bull-Mastif -0.7526  0.5469
    Caniche      0.9123 -0.0162
    Chihuahua    0.8408  0.8439
    Cocker       0.7333  0.0791
    Colley      -0.1173 -0.5261
    Variables (the 10 first):
                Can1    Can2
    Taille+   0.8511 -1.2317
    Taille++ -0.8367 -0.0206
    Taille-   1.1850  0.9239
    Poids+   -0.3054 -0.8189
    Poids++  -1.0151  0.9739
    Poids-    1.1689  0.8243
    Veloc+    0.6037 -0.8878
    Veloc++  -0.8921 -0.3718
    Veloc-    0.3199  1.0449
    Intell+   0.3694 -0.2855

    and factor analysis of mixed data (FAMD).
    
    >>> #FAMD
    >>> D = load_heart("subset")
    >>> X = D.drop(columns=["disease"])
    >>> clf = GFA()
    >>> clf.fit(X,y)
    GFA()
    >>> summaryGFA(clf)
                        General Factor Analysis - Results
    Eigenvalues informations:
                             Can1     Can2     Can3     Can4  ...     Can7     Can8     Can9     Can10
    Variance               2.3749   1.1618   1.0641   1.0243  ...   0.7950   0.7223   0.4739    0.4363
    Difference             1.2131   0.0978   0.0398   0.0441  ...   0.0727   0.2483   0.0376       NaN
    % of var.             23.7495  11.6184  10.6408  10.2431  ...   7.9497   7.2227   4.7392    4.3629
    Cumulative % of var.  23.7495  35.3678  46.0086  56.2517  ...  83.6751  90.8979  95.6371  100.0000
    [4 rows x 10 columns]
    Individuals (the 10 first):
         Can1    Can2
    0 -1.0697 -0.9664
    1  0.8760 -0.8987
    2 -0.6487  3.0729
    3 -0.8548  1.0046
    4  0.2919 -0.3426
    5  0.2375 -0.2891
    6  2.2459  5.0207
    7  2.3149  0.3848
    8  0.7471 -0.1014
    9  2.8908 -0.2045
    Variables (the 10 first):
                       Can1    Can2
    age              0.6420  0.3410
    restbpress       0.4544  0.2595
    max_hrate       -0.7802  0.0723
    asympt           1.0310 -0.4093
    atyp_angina     -1.0302 -0.0652
    non_anginal     -0.8281  1.1019
    typ_angina      -1.3976  1.0534
    f               -0.1198 -0.1910
    t                1.4456  2.3035
    left_vent_hyper -1.2533  3.6476
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if self is an object of class GFA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "gfa":
        raise ValueError("'self' must be an object of class GFA")

    print("                     General Factor Analysis - Results                     ")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #eigenvalues informations
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nEigenvalues informations:")
    eig = obj.eig_.T.round(decimals=digits)
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
    ind_coord = ind_coord.iloc[:nb_element,:ncp].round(decimals=digits)
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
    var_coord = var_coord.iloc[:nb_element,:ncp].round(decimals=digits)
    if to_markdown:
        var_coord = var_coord.to_markdown(tablefmt=tablefmt,**kwargs)
    print(var_coord)