# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pyreadr 
import pathlib


DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"

def load_vins():
    # Load Data
    wine = pd.read_excel(DATASETS_DIR/"vin_bordelais.xls",header=0,index_col=0)
    return wine.set_index("Annee")

def load_wine():
    # Load Data
    wine = pd.read_excel(DATASETS_DIR/"wine_bordelais.xls",header=0)
    return wine