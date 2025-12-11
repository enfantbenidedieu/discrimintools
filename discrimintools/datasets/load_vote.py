# -*- coding: utf-8 -*-
from pandas import read_excel
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

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
    >>> from discrimintools import GFALDA
    >>> D = load_vote("subset") # load subset data
    >>> y, X = D["group"], D.drop(columns=["group"]) # split into X and y
    >>> clf = GFALDA()
    >>> clf.fit(X,y)
    GFALDA()
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