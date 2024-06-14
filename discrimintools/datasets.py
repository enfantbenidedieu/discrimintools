# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pyreadr 
import pathlib

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"

def load_vins():
    """
    
    References
    ----------
    """
    # Load Data
    wine = pd.read_excel(DATASETS_DIR/"vin_bordelais.xls",header=0,index_col=0)
    return wine.set_index("Annee")

def load_wine():
    # Load Data
    wine = pd.read_excel(DATASETS_DIR/"wine_bordelais.xls",header=0)
    return wine

def load_olive_oil(which="active"):
    """
    Olive oil
    ---------

    Parameters
    ----------
    `which` : string specifying either active or supplementary set

    Returns
    -------
    """
    if which not in ["active","sup"]:
        raise ValueError("'which' should be one of 'active', 'sup'")
    
    if which == "active":
        oil = pd.read_excel(DATASETS_DIR/"Olive_Oil_Candisc.xlsx",sheet_name="dataset")
    elif which == "sup":
        oil = pd.read_excel(DATASETS_DIR/"Olive_Oil_Candisc.xlsx",sheet_name="supplementaires")
    return oil

def load_wine():
    """
    Wine: Chemical composition of three cultivars of wine
    -----------------------------------------------------

    For more, see https://www.rdocumentation.org/packages/candisc/versions/0.9.0/topics/Wine
    """
    wine = pyreadr.read_r(DATASETS_DIR/"Wine.RData")["Wine"]
    return wine

def load_wines_disca():
    """
    Wines dataset
    -------------

    Notes
    -----
    12 wines coming from 3 diferent origins (4 wines per origin)

    References
    ----------
    Abdi, H. (2007). Discriminant correspondence analysis. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics. Thousand Oaks (CA): Sage. pp. 270-275.
    """
    wines = pd.read_excel(DATASETS_DIR/"wines_disca.xlsx",header=0,index_col=0)
    return wines

def load_canines():
    """
    Races canines
    -------------
    """
    canines = pd.read_excel(DATASETS_DIR/"canines.xls",header=0,index_col=0)
    return canines

def load_mushroom(which="train"):
    """
    Mushroom dataset
    ----------------
    
    """
    if which not in ["train","test"]:
        raise ValueError("'which' should be one of 'train', 'test'")
    # Load all file
    df = pd.read_excel(DATASETS_DIR/"mushroom.xls")
    if which == "train":
        df = df.loc[df.SAMPLE_STATUS == 'train'].drop('SAMPLE_STATUS',axis='columns')
    elif which == "test":
        df = df.loc[df.SAMPLE_STATUS == 'test'].drop('SAMPLE_STATUS',axis='columns')
    return df

def load_congress_vote(which="train"):
    """
    
    """
    if which not in ["train","test","subset"]:
        raise ValueError("'which' should be one of 'train', 'test', 'subset'")

    if which == "train":
        vote = pd.read_excel(DATASETS_DIR/"CongressVotePipeline.xlsx",sheet_name="train",header=0)
    elif which == "test":
        vote = pd.read_excel(DATASETS_DIR/"CongressVotePipeline.xlsx",sheet_name="test",header=0)
    elif which == "subset":
        vote = pd.read_excel(DATASETS_DIR/"CongressVotePipeline.xlsx",sheet_name="subset",header=0)
    return vote

def load_heart():
    """
    Heart
    -----
    
    """
    heart = pd.read_excel(DATASETS_DIR/"heart_weka_only_male.xls")
    return heart
