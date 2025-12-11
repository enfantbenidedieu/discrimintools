# -*- coding: utf-8 -*-
from pandas import read_excel, read_csv
from pathlib import Path

#set directory
DATASETS_DIR = Path(__file__).parent / "data"

def load_dataset(name="iris"):
    """
    Load an example dataset

    Note
    ----
    This function provides quick access to a small number of example datasets that are useful.

    Parameters
    ----------
    name : str, default = 'iris'
        The name of the dataset. Possible values are:

        - 'breast' for the Breat Cancer dataset.
        - 'cultivar' for the Chemical composition of three cultivars of wine.
        - 'fish' for the fish dataset
        - 'iris' for the iris dataset
        - 'sonar' for the sonar dataset
        - 'wavenoise' for the wavenoise dataset

    Returns
    -------
    data : DataFrame of shape (n_samples, n_columns)
        The dataset loaded.
    """
    if name == "breast":
        data = read_excel(DATASETS_DIR/"breast_cancer.xlsx",header=0,sheet_name=0).rename(columns={"class" : "Class"})
        data.__doc__ = """
        Breat Cancer dataset

        """
    elif name == "cultivar":
        data = read_excel(DATASETS_DIR/"cultivar.xlsx",header=0,sheet_name=0)
        data.__doc__= """
        Wine: Chemical composition of three cultivars of wine

        For more, see https://archive.ics.uci.edu/dataset/109/wine

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import CANDISC
        >>> D = load_dataset("cultivar") # load data
        >>> y, X = D["Cultivar"], D.drop(columns=["Cultivar"]) # split into X and y
        >>> clf = CANDISC()
        >>> clf.fit(X,y)
        CANDISC()
        """
    elif name == "fish":
        data = read_excel(DATASETS_DIR/"fish.xlsx",header=0,index_col=0)
        data.__doc__ = """
        Fish dataset

        Note
        ----
        The data in this example are measurements of 159 fish caught in Finland's Lake Laengelmaevesi; this data set
        is available from the Puranen. For each of the seven species (bream, roach, whitefish, parkki, perch, pike, and
        smelt), the weight, length, height, and width of each fish are tallied. Three different length measurements are
        recorded: from the nose of the fish to the beginning of its tail, from the nose to the notch of its tail, and from
        the nose to the end of its tail. The height and width are recorded as percentages of the third length variable.

        """
    elif name == "iris":
        data = read_csv(DATASETS_DIR/"iris.csv",header=0,sep=",")
        data.__doc__ = """
        Iris dataset

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import CANDISC
        >>> D = load_dataset("iris") # load data
        >>> y, X = DTrain["Species"], D.drop(columns=["Species"]) # split into X and y
        >>> clf = CANDISC()
        >>> clf.fit(XTrain,yTrain)
        CANDISC()
        """
    elif name == "sonar":
        data = read_excel(DATASETS_DIR/"sonar.xlsx",sheet_name="Feuil1",header=0,index_col=None)
        data.__doc__ = """
        Sonar dataset

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import DISCRIM, STEPDISC
        >>> DTrain = load_dataset("sonar") # load data
        >>> yTrain, XTrain = DTrain["Class"], DTrain.drop(columns=["Class"]) # split into X and y
        >>> #linear discriminant analysis (LDA)
        >>> clf = DISCRIM()
        >>> clf.fit(XTrain,yTrain)
        DISCRIM(priors="prop")
        >>> #stepwise discriminant analysis (STEPDISC)
        >>> clf2 = STEPDISC(method="backward")
        >>> clf2.fit(clf)
        STEPDISC(method="backward")
        """
    elif name == "wavenoise":
        data = read_excel(DATASETS_DIR/"wavenoise.xlsx",sheet_name="Feuil1",header=0,index_col=None)
        data.__doc__ = """
        Wave Noise dataset

        Examples
        --------
        >>> from discrimintools.datasets import load_dataset
        >>> from discrimintools import DISCRIM, STEPDISC
        >>> D = load_dataset("wavenoise") # load data
        >>> y, X = D["classe"], D.drop(columns=["classe"]) # split into X and y
        >>> #linear discriminant analysis (LDA)
        >>> clf = DISCRIM()
        >>> clf.fit(X,y)
        DISCRIM(priors='prop')
        >>> #stepwise discriminant analysis (STEPDISC)
        >>> clf2 = STEPDISC(method="backward")
        >>> clf2.fit(clf)
        STEPDISC(method="backward")
        """
    else:
        raise ValueError("{} not supported".format(name))
    
    return data