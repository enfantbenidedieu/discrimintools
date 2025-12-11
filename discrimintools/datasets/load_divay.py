# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    [1] Hervé Abdi (2007). `Discriminant correspondence analysis <https://personal.utdallas.edu/~herve/Abdi-DCA2007-pretty.pdf>`_. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics. Thousand Oaks (CA): Sage. pp. 270-275.

    [2] Ricco Rakotomalala (2012), `Analyse des correspondances discriminante <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Discriminant_Correspondence_Analysis.pdf>`_, Université Lumière Lyon 2.

    [3] Ricco Rakotomalala (2020), `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_, Version 1.0, Université Lumière Lyon 2.
    
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