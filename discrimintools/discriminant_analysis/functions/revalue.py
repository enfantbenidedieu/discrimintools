# -*- coding: utf-8 -*-
from pandas import Series, Categorical

#intern function
from .utils import check_is_dataframe

def revalue(X):
    """
    Revalue Categoricals Variables

    Check if two categoricals variables have same levels and replace with new values

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns) or Series of shape (n_samples,)
        X contains categoricals variables.

    Return(s)
    ---------
     Y : DataFrame of shape (n_samples, n_columns)
        Revaluate columns
    """
    if isinstance(X,Series): #convert to DataFrame if X is a pandas Series
        X = X.to_frame()
    
    #check if X is an instance of class pd.DataFrame
    check_is_dataframe(X=X)

    #check if shape greater than 1:
    Y = X.copy()
    if Y.shape[1]>1:
        for i in range(X.shape[1]-1):
            for j in range(i+1,X.shape[1]):
                if (X.iloc[:,i].dtype in ["object","category"]) and (X.iloc[:,j].dtype in ["object","category"]):
                    intersect = list(set(X.iloc[:,i].dropna().unique().tolist()) & set(X.iloc[:,j].dropna().unique().tolist()))
                    if len(intersect)>=1:
                        valuei = {x : X.columns.tolist()[i]+"_"+str(x) for x in X.iloc[:,i].dropna().unique().tolist()}
                        valuej = {x : X.columns.tolist()[j]+"_"+str(x) for x in X.iloc[:,j].dropna().unique().tolist()}
                        Y.iloc[:,i], Y.iloc[:,j] = X.iloc[:,i].map(valuei), X.iloc[:,j].map(valuej)

    #convert to categorical
    for q in Y.columns:
        if Y[q].dtype in ["object","category"]:
            Y[q] = Categorical(Y[q],categories=sorted(Y[q].dropna().unique().tolist()),ordered=True)
    return Y