# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    >>> from discrimintools import DISCRIM
    >>> D = load_infidelity("train") # load training data
    >>> y, X = D["Infidelity"], D.drop(columns=["Infidelity"]) # split into X and y
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    DISCRIM(priors='prop')
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