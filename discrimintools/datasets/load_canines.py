# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    [1] Michel Tenenhaus (1996), « Méthodes statistiques en gestion », Dunod.

    [2] Michel Tenenhaus (2007), « Statistique - Méthodes pour décrire, expliquer et prévoir », Dunod.

    [3] Ricco Rakotomalala (2008), « `AFCM - Races canines <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Acm.pdf>`_ ».

    [4] Ricco Rakotomalala (2009), « `Analyse des Correspondances Multiple avec R <https://eric.univ-lyon2.fr/ricco/cours/didacticiels/R/afcm_avec_r.pdf>`_ ».

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