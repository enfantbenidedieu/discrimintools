# -*- coding: utf-8 -*-
from pandas import DataFrame, concat, get_dummies, CategoricalDtype
from pandas.api.types import is_numeric_dtype, is_string_dtype

#intern function
from .utils import check_is_dataframe

def model_matrix(X:DataFrame) -> DataFrame:
    """
    Model Matrix - Construct Design Matrices

    Description
    -----------
    Create a design (model) matrix. If variable is categoric, it create a disjuntive table without the first category.

    Parameters
    ----------
    X : pandas DataFrame of shape (n_samples, n_columns)
        Input data

    Return
    ------
    data : pandas DataFrame of shape (n_samples, n_new_columns)
        Output data
    
    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)

    def recode(x):
        if is_numeric_dtype(x):
            return x
        elif is_string_dtype(x):
            uq_x = sorted(x.unique().tolist())
            x = x.astype(CategoricalDtype(categories=uq_x,ordered=True))
            return get_dummies(x,prefix=x.name,prefix_sep="",drop_first=True,dtype=int)
        else:
            raise TypeError("{} is not supported.".format(type(x)))
    
    data = concat((recode(x=X[k]) for k in X.columns),axis=1)
    return data



