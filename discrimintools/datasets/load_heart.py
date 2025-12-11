# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    >>> from discrimintools import GFALDA
    >>> D = load_heart("subset") # load subset data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = GFALDA(n_components=5)
    >>> clf.fit(X,y)
    GFALDA(n_components=5)
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