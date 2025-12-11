# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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