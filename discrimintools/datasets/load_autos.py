# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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