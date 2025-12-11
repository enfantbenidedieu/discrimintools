# -*- coding: utf-8 -*-
from numpy import zeros
from pandas import Series, DataFrame, concat, get_dummies

def tab_disjunctive(
        X,dummies_cols=None,prefix=False,sep="_"
):
    """
    Convert categorical variable into dummy/indicator variables

    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Input data.

    dummies_cols : None or list, default = None
        Columns from orginal disjontive table.

    prefix : bool, default = False
        If True, append DataFrame with columns names.

    sep : str, default = "_"
        If appending prefix, separator/delimiter to use.

    Returns
    -------
    DataFrame :
        Dummy-coded data.
    """
    #convert to DataFrame if Series
    if isinstance(X,Series):
        X = X.to_frame()

    def set_dummies(
            x, prefix = False, sep = "_"
    ):
        """
        Binary coding

        Parameters
        ----------
        x : Series of shape (n_samples,)
            Input data.

        prefix : bool, default = False
            If True, append DataFrame with columns names.

        sep : str, default = "_"
            If appending prefix, separator/delimiter to use.

        Returns
        -------
        DataFrame :
            Dummy-coded data.
        """
        #add prefix
        if prefix:
            return get_dummies(x,prefix=x.name,prefix_sep=sep,dtype=int)
        else:
            return get_dummies(x,dtype=int)

    #dummies
    dummies = concat((set_dummies(X[j],prefix=prefix,sep=sep) for j in list(X.columns)),axis=1)
    #update if dummies_cols is not None
    if dummies_cols is not None:
        #initialize
        dummies_all = DataFrame(zeros((X.shape[0],len(dummies_cols))),index=X.index,columns=dummies_cols)
        #update with dummies in new individuals
        if len(dummies_cols) >= dummies.shape[1]:
            dummies_all.loc[:,list(dummies.columns)] = dummies
        else:
            dummies_all.loc[:,dummies_cols] = dummies.loc[:,dummies_cols]
    else:
        dummies_all = dummies.copy()
    return dummies_all 