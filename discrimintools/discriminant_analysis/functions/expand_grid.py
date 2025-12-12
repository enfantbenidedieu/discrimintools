# -*- coding: utf-8 -*-
from itertools import product
from pandas import DataFrame, CategoricalDtype

def expand_grid(itrs, strings_as_category=True):
    """
    expand_grid

    Create a DataFrame from all combinations of the supplied vectors or factors

    Parameters
    ----------
    itrs : list, tuple or a dict
        These

    string_as_category : bool
        if string columns are converted to category.

    Returns
    -------
    X : DataFrame
        One row for each combination of the supplied category

    Examples
    --------
    >>> from discrimintools import expand_grid
    >>> itrs = dict(color=['red','green','blue'],cylinders=[6, 8],vehicle=['car','van','truck'])
    >>> expand_grid(itrs)
                color  cylinders vehicle
            0     red          6     car
            1     red          6     van
            2     red          6   truck
            3     red          8     car
            4     red          8     van
            5     red          8   truck
            6   green          6     car
            7   green          6     van
            8   green          6   truck
            9   green          8     car
            10  green          8     van
            11  green          8   truck
            12   blue          6     car
            13   blue          6     van
            14   blue          6   truck
            15   blue          8     car
            16   blue          8     van
            17   blue          8   truck
    """
    if isinstance(itrs,dict):
        X = DataFrame([x for x in product(*itrs.values())], columns=itrs.keys())
    elif isinstance(itrs, (list,type)):
        X = DataFrame([x for x in product(*itrs)],columns=["Var"+str(i+1) for i in range(len(itrs))])
    else:
        raise TypeError(f"{type(itrs)} is not supported.")
    
    #convert string to category
    if strings_as_category:
        for k in X.columns:
            if X[k].dtype in ["object","category"]:
                classes = sorted(X[k].unique().tolist())
                X[k] = X[k].astype(CategoricalDtype(categories=classes,ordered=True))
    return X