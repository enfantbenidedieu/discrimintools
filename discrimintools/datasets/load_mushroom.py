# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    >>> from discrimintools import GFALDA
    >>> D = load_mushroom("train") # load training data
    >>> y, X = D["classe"], D.drop(columns=["classe"]) # split into X and y
    >>> clf = GFALDA(n_components=5)
    >>> clf.fit(X,y)
    GFALDA(n_components=5)
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