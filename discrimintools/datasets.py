# -*- coding: utf-8 -*-
from __future__ import annotations

from pandas import read_excel, read_csv
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"

def load_alcools(element="train"):
    """
    Alcools dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    alcools : DataFrame of shape (n_samples, n_columns)
        The alcools dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_alcools
    >>> from discrimintools import DISCRIM
    >>> D = load_alcools("train") # load training data
    >>> y, X = D["TYPE"], D.drop(columns=["TYPE"]) # split into X and y
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    DISCRIM(priors='prop')
    """
    if element == "train":
        alcools = read_excel(DATASETS_DIR/"alcools.xlsx",header=0,sheet_name="Feuil1")
    elif element == "test":
        alcools = read_excel(DATASETS_DIR/"alcools.xlsx",header=0,sheet_name="Feuil2")
    else:
        raise ValueError("'element' should be one of 'train', 'test'.")
    #set cocumentation
    alcools.__doc__ = """
    Alcools dataset

    """
    return alcools

def load_autos(element="train"):
    """
    Autos dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    autos : DataFrame of shape (n_samples, n_columns)
        The autos dataset.

    References
    ----------
    [1] Saporta Gilbert (2011), « `Probabilités, Analyse des données et Statistiques <https://en.pdfdrive.to/dl/probabilites-analyses-des-donnees-et-statistiques>`_ », Editions TECHNIP, 3ed.

    [2] Ricco Rakotomalala (2020), « `Pratique Des Méthodes Factorielles avec Python <https://hal.science/hal-04868625v1/document>`_ », Version 1.0, Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_autos
    >>> from discrimintools import DISCRIM
    >>> D = load_autos("train") # load training data
    >>> y, X = D["Finition"], D.drop(columns=["Finition"]) # split into X and y
    >>> clf = DISCRIM(classes=("M","B","TB"))
    >>> clf.fit(X,y)
    DISCRIM(priors='prop',classes=('M','B','TB'))
    """
    if element == "train":
        autos = read_excel(DATASETS_DIR/"autos.xlsx",header=0,index_col=0,sheet_name="Feuil1")
    elif element == "test":
        autos = read_excel(DATASETS_DIR/"autos.xlsx",header=0,index_col=0,sheet_name="Feuil2")
    else:
        raise ValueError("'element' should be one of 'train', 'test'.")
    #set cocumentation
    autos.__doc__ = """
    Autos dataset

    """
    return autos

def load_canines(element="train"):
    """
    Canines dataset
    
    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    canines : DataFrame of shape (n_samples, n_columns)
        The canines dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2008), « `AFCM - Races canines <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Acm.pdf>`_ ».

    [2] Ricco Rakotomalala (2009), « `Analyse des Correspondances Multiple avec R <https://eric.univ-lyon2.fr/ricco/cours/didacticiels/R/afcm_avec_r.pdf>`_ ».

    [3] Tenenhaus Michel (1996), « Méthodes statistiques en gestion », Dunod.

    [4] Tenenhaus Michel (2007), « Statistique - Méthodes pour décrire, expliquer et prévoir », Dunod.

    Examples
    --------
    >>> from discrimintools.datasets import load_canines
    >>> from discrimintools import DiCA
    >>> D = load_canines("train") # load training data
    >>> y, X = D["Fonction"], D.drop(columns=["Fonction"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    """
    if element == "train":
        canines = read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil1",header=0,index_col=0)
    elif element == "test":
        canines = read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil2",header=0,index_col=0)
    else:
        raise ValueError("'element' should be one of 'train' or 'test'.")
    #set cocumentation
    canines.__doc__ = """
    Canines dataset

    """
    return canines

def load_dataset(name="iris"):
    """
    Load an example dataset

    Note
    ----
    This function provides quick access to a small number of example datasets that are useful.

    Parameters
    ----------
    name : str, default = 'iris'
        The name of the dataset. Possible values are:

        - 'breast' for the Breat Cancer dataset.
        - 'cultivar' for the Chemical composition of three cultivars of wine.
        - 'fish' for the fish dataset
        - 'iris' for the iris dataset
        - 'sonar' for the sonar dataset
        - 'wavenoise' for the wavenoise dataset

    Returns
    -------
    data : DataFrame of shape (n_samples, n_columns)
        The dataset loaded.
    """
    if name == "breast":
        data = read_excel(DATASETS_DIR/"breast_cancer.xlsx",header=0,sheet_name=0).rename(columns={"class" : "Class"})
        data.__doc__ = """
        Breat Cancer dataset

        """
    elif name == "cultivar":
        data = read_excel(DATASETS_DIR/"cultivar.xlsx",header=0,sheet_name=0)
        data.__doc__= """
        Wine: Chemical composition of three cultivars of wine

        For more, see https://archive.ics.uci.edu/dataset/109/wine

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import CANDISC
        >>> D = load_dataset("cultivar") # load data
        >>> y, X = D["Cultivar"], D.drop(columns=["Cultivar"]) # split into X and y
        >>> clf = CANDISC()
        >>> clf.fit(X,y)
        CANDISC()
        """
    elif name == "fish":
        data = read_excel(DATASETS_DIR/"fish.xlsx",header=0,index_col=0)
        data.__doc__ = """
        Fish dataset

        Note
        ----
        The data in this example are measurements of 159 fish caught in Finland's Lake Laengelmaevesi; this data set
        is available from the Puranen. For each of the seven species (bream, roach, whitefish, parkki, perch, pike, and
        smelt), the weight, length, height, and width of each fish are tallied. Three different length measurements are
        recorded: from the nose of the fish to the beginning of its tail, from the nose to the notch of its tail, and from
        the nose to the end of its tail. The height and width are recorded as percentages of the third length variable.

        """
    elif name == "iris":
        data = read_csv(DATASETS_DIR/"iris.csv",header=0,sep=",")
        data.__doc__ = """
        Iris dataset

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import CANDISC
        >>> D = load_dataset("iris") # load data
        >>> y, X = DTrain["Species"], D.drop(columns=["Species"]) # split into X and y
        >>> clf = CANDISC()
        >>> clf.fit(XTrain,yTrain)
        CANDISC()
        """
    elif name == "sonar":
        data = read_excel(DATASETS_DIR/"sonar.xlsx",sheet_name="Feuil1",header=0,index_col=None)
        data.__doc__ = """
        Sonar dataset

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import DISCRIM, STEPDISC
        >>> DTrain = load_dataset("sonar") # load data
        >>> yTrain, XTrain = DTrain["Class"], DTrain.drop(columns=["Class"]) # split into X and y
        >>> #linear discriminant analysis (LDA)
        >>> clf = DISCRIM()
        >>> clf.fit(XTrain,yTrain)
        DISCRIM(priors="prop")
        >>> #stepwise discriminant analysis (STEPDISC)
        >>> clf2 = STEPDISC(method="backward")
        >>> clf2.fit(clf)
        STEPDISC(method="backward")
        """
    elif name == "wavenoise":
        data = read_excel(DATASETS_DIR/"wavenoise.xlsx",sheet_name="Feuil1",header=0,index_col=None)
        data.__doc__ = """
        Wave Noise dataset

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import DISCRIM, STEPDISC
        >>> D = load_dataset("wavenoise") # load data
        >>> y, X = D["classe"], D.drop(columns=["classe"]) # split into X and y
        >>> #linear discriminant analysis (LDA)
        >>> clf = DISCRIM()
        >>> clf.fit(X,y)
        DISCRIM(priors='prop')
        >>> #stepwise discriminant analysis (STEPDISC)
        >>> clf2 = STEPDISC(method="backward")
        >>> clf2.fit(clf)
        STEPDISC(method="backward")
        """
    else:
        raise ValueError("{} not supported".format(name))
    
    return data

def load_divay(element = "train"):
    """
    Divay dataset

    Note
    ----
    12 wines coming from 3 diferent origins (4 wines per origin).

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    divay : DataFrame of shape (n_samples, n_columns) 
        The divay dataset.

    References
    ----------
    [1] Abdi, H. (2007). `Discriminant correspondence analysis <https://personal.utdallas.edu/~herve/Abdi-DCA2007-pretty.pdf>`_. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics. Thousand Oaks (CA): Sage. pp. 270-275.

    [2] Ricco Rakotomalala (2012), `Analyse des correspondances discriminante <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Discriminant_Correspondence_Analysis.pdf>`_,

    [3] Ricco Rakotomalala (2020), `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_, Version 1.0.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA
    >>> D = load_divay() # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    """
    if element == "train":
        divay = read_excel(DATASETS_DIR/"divay.xlsx",sheet_name="Feuil1",header=0,index_col=None)
    elif element == "test":
        divay = read_excel(DATASETS_DIR/"divay.xlsx",sheet_name="Feuil2",header=0,index_col=None)
    else:
        raise ValueError("'element' should be one of 'train', 'test'")
    #set cocumentation
    divay.__doc__ = """
    Divay dataset

    """
    return divay

def load_heart(element = "subset"):
    """
    Heart dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'subset' for subset of all dataset (with few columns).
        - 'train' for training dataset. 
        - 'test' for testing dataset.
    
    Returns
    -------
    heart : DataFrame of shape (n_samples, n_columns)
        The heart dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import LDAPC
    >>> D = load_heart("subset") # load subset data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = LDAPC(n_components=5)
    >>> clf.fit(X,y)
    LDAPC()
    """
    if element == "train":
        heart = read_excel(DATASETS_DIR/"heart.xlsx",sheet_name="Feuil1",header=0).query("status == 'train'").drop(columns=["status"])
    elif element == "test":
        heart = read_excel(DATASETS_DIR/"heart.xlsx",sheet_name="Feuil1",header=0).query("status == 'test'").drop(columns=["status"])
    elif element == "subset":
        heart = read_excel(DATASETS_DIR/"heart.xlsx",sheet_name="Feuil2",header=0)
    else:
        raise ValueError("'element' should be one of 'train', 'test', 'subset'")
    #set documentation
    heart.__doc__ = """
    Heart dataset

    """
    return heart

def load_infidelity(element="train"):
    """
    Infidelity dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    infidelity : DataFrame of shape (n_samples, n_columns)
        The infidelity dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Régression logistique sous Python <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Python_Regression_Logistique.pdf>`_ », Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_infidelity
    >>> from discrimintools import PLSGLM
    >>> D = load_infidelity("train") # load training data
    >>> y, X = D["Infidelity"], D.drop(columns=["Infidelity"]) # split into X and y
    >>> clf = PLSGLM()
    >>> clf.fit(X,y)
    PLSGLM()
    """
    if element == "train":
        infidelity = read_excel(DATASETS_DIR/"infidelity.xlsx",header=0,sheet_name="Feuil1")
    elif element == "test":
        infidelity = read_excel(DATASETS_DIR/"infidelity.xlsx",header=0,sheet_name="Feuil2")
    else:
        raise ValueError("'element' should be one of 'train', 'test'.")
    #set cocumentation
    infidelity.__doc__ = """
    Infidelity dataset

    """
    return infidelity

def load_mushroom(element="train"):
    """
    Mushroom dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    mushroom : DataFrame of shape (n_samples, n_columns) 
        The mushroom dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from discrimintools.dataset import load_mushroom
    >>> from discrimintools import LDAPC
    >>> D = load_mushroom("train") # load training data
    >>> y, X = D["classe"], D.drop(columns=["classe"]) # split into X and y
    >>> clf = LDAPC(n_components=5)
    >>> clf.fit(X,y)
    LDAPC()
    """
    if element == "train":
        mushroom = read_excel(DATASETS_DIR/"mushroom.xlsx",sheet_name="Feuil1",header=0,index_col=0)
    elif element == "test":
        mushroom = read_excel(DATASETS_DIR/"mushroom.xlsx",sheet_name="Feuil2",header=0,index_col=0)
    else:
        raise ValueError("'element' should be one of 'train', 'test'")
    #set documentation
    mushroom.__doc__ = """
    Mushroom dataset

    """
    return mushroom

def load_oliveoil(element="train"):
    """
    Olive oil dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    oil : DataFrame of shape (n_samples, n_columns) 
        The olive oil dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_oliveoil
    >>> from discrimintools import CANDISC
    >>> D = load_oliveoil("train") # load training data
    >>> y, X = D["CLASSE"], D.drop(columns=["CLASSE"]) # split into X and y
    >>> clf = CANDISC(n_components=2)
    >>> clf.fit(X,y)
    CANDISC()
    """
    if element == "train":
        oil = read_excel(DATASETS_DIR/"oliveoil.xlsx",sheet_name="Feuil1",header=0,index_col=None)
    elif element == "test":
        oil = read_excel(DATASETS_DIR/"oliveoil.xlsx",sheet_name="Feuil2",header=0,index_col=None)
    else:
        raise ValueError("'element' should be one of 'train', 'test'")
    #set documentation
    oil.__doc__ = """
    Olive oil dataset

    """
    return oil

def load_vins(element="train"):
    """
    Vins de Bordeaux dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    vins : DataFrame of shape (n_samples, n_columns) 
        The vins de Bordeaux dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2008), « `Analyse discriminante descriptive - vins de Bordeaux <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Canonical_Discriminant_Analysis.pdf>`_ ».

    [3] Ricco Rakotomalala (2011), « `Analyse factorielle discriminante - Diaporama <https://eric.univ-lyon2.fr/ricco/cours/slides/analyse_discriminante_descriptive.pdf>`_ ».

    [3] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.

    [4] Tenenhaus Michel (1996), « Méthodes statistiques en gestion », Dunod.

    Examples
    --------
    >>> from discrimintools.datasets import load_vins
    >>> from discrimintools import CANDISC
    >>> D = load_vins("train") # load training data
    >>> y, X = D["Qualite"], D.drop(columns=["Qualite"]) # split into X et y
    >>> clf = CANDISC(classes=("Mediocre","Moyen","Bon"))
    >>> clf.fit(X,y)
    CANDISC(classes=("Mediocre","Moyen","Bon"))
    """
    if element == "train":
        vins = read_excel(DATASETS_DIR/"vins.xlsx",header=0,index_col=0,sheet_name="Feuil1")
    elif element == "test":
        vins = read_excel(DATASETS_DIR/"vins.xlsx",header=0,index_col=0,sheet_name="Feuil2")
    else:
        raise ValueError("'element' should be one of 'train', 'test'.")
    vins.__doc__ = """
    Vins bordelais dataset

    """
    return vins

def load_vote(element="train"):
    """
    Congressional Voting Records dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:

        - 'subset' for subset of all dataset (with few columns).
        - 'train' for training dataset. 
        - 'test' for testing dataset.
    
    Returns
    -------
    vote : DataFrame of shape (n_samples, n_columns)
        The congressional voting records dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.

    Examples
    --------
    >>> from discrimintools.datasets import load_vote
    >>> from discrimintools import LDAPC
    >>> D = load_vote("subset") # load subset data
    >>> y, X = D["group"], D.drop(columns=["group"]) # split into X and y
    >>> clf = LDAPC()
    >>> clf.fit(X,y)
    LDAPC()
    """
    if element == "train":
        vote = read_excel(DATASETS_DIR/"vote.xlsx",sheet_name="Feuil1",header=0,index_col=0)
    elif element == "test":
        vote = read_excel(DATASETS_DIR/"vote.xlsx",sheet_name="Feuil2",header=0,index_col=0)
    elif element == "subset":
        vote = read_excel(DATASETS_DIR/"vote.xlsx",sheet_name="Feuil3",header=0,index_col=0)
    else:
        raise ValueError("'element' should be one of 'train', 'test', 'subset'")
    #set documentation
    vote.__doc__ = """
    Congressional Voting Records dataset

    """
    return vote

def load_wine(element="train"):
    """
    Bordeaux Wine dataset

    Parameters
    ----------
    element : str, default = 'train'
        The dataset to load. Possible values are:
        
        - 'train' for training dataset. 
        - 'test' for testing dataset.

    Returns
    -------
    wine : DataFrame of shape (n_samples, n_columns) 
        The Bordeaux wine dataset.

    References
    ----------
    [1] Ricco Rakotomalala (2008), « `Analyse discriminante descriptive - vins de Bordeaux <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Canonical_Discriminant_Analysis.pdf>`_ ».

    [2] Ricco Rakotomalala (2011), « `Analyse factorielle discriminante - Diaporama <https://eric.univ-lyon2.fr/ricco/cours/slides/analyse_discriminante_descriptive.pdf>`_ ».

    [3] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2..

    [4] Tenenhaus Michel (1996), « Méthodes statistiques en gestion », Dunod.
    
    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC
    >>> D = load_wine("train") # load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC(classes=("bad","medium","good"))
    >>> clf.fit(X,y)
    CANDISC(classes=("bad","medium","good"))
    """
    if element == "train":
        wine = read_excel(DATASETS_DIR/"wine.xlsx",header=0,index_col=0,sheet_name="Feuil1")
    elif element == "test":
        wine = read_excel(DATASETS_DIR/"wine.xlsx",header=0,index_col=0,sheet_name="Feuil2")
    else:
        raise ValueError("'element' should be one of 'train', 'test'.")
    wine.__doc__ = """
    Bordeaux Wine dataset

    """
    return wine