# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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