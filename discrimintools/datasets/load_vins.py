# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    [1] Michel Tenenhaus (1996), « Méthodes statistiques en gestion », Dunod.

    [2] Ricco Rakotomalala (2008), « `Analyse discriminante descriptive - vins de Bordeaux <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Canonical_Discriminant_Analysis.pdf>`_ », Université Lumière Lyon 2.
    
    [3] Ricco Rakotomalala (2011), « `Analyse factorielle discriminante - Diaporama <https://eric.univ-lyon2.fr/ricco/cours/slides/analyse_discriminante_descriptive.pdf>`_ », Université Lumière Lyon 2.

    [4] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Version 1.0, Université Lumière Lyon 2.

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