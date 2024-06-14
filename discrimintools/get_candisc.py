# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_candisc_ind(self):
    """
    Extract the results for individuals - CANDISC
    ---------------------------------------------

    Description
    -----------
    Extract the results (factor coordinates) for active individuals from Canonical Discriminant Analysis (CANDISC) outputs.

    Usage
    ````python
    >>> get_candisc_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class CANDISC

    Returns
    -------
    dictionary of dataframes containing all the results for the active individuals including:
    
    `coord` : factor coordinates for the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # Load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import CANDISC, get_candisc_ind
    >>> res_candisc = CANDISC(n_components=2,target=["species"],priors="prop",parallelize=False)
    >>> res_candisc.fit(iris)
    >>> ind = get_candisc_ind(res_candisc)
    ```
    """
    if self.model_ != "candisc":
        raise TypeError("'self' must be an object of class CANDISC.")
    return self.ind_

def get_candisc_var(self,choice="correlation"):
    """
    Extract the results for variables - CANDISC
    -------------------------------------------

    Description
    -----------
    Extract the results (covariance, canonical correlation) for variables from Canonical Discriminant Analysis (CANDISC) outputs.

    Usage
    -----
    ```python
    >>> get_candisc_var(self,choice=("correlation","covariance"))
    ```

    Parameters
    ----------
    `self` : an object of class CANDISC

    `choice` : the element to subset from the output. Allowed values are :
        * "correlation" for canonical correlation (total, withon, between)
        * "covariance" for covariance (total, within, between)

    Returns
    -------
    dictionary of dataframes containing all the results for the variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import CANDISC, get_candisc_var
    >>> res_candisc = CANDISC(n_components=2,target=["species"],priors="prop",parallelize=False)
    >>> res_candisc.fit(iris)
    >>> # results for canonical correlation
    >>> corr = get_candisc_var(res_candisc,choice="correlation") 
    >>> # results for covariance
    >>> covar = get_candisc(res_candisc,choice="covariance") 
    ```
    """
    if self.model_ != "candisc":
        raise TypeError("'self' must be an object of class CANDISC")
    
    if choice not in ["correlation","covariance"]:
        raise ValueError("'choice' should be one of 'correlation', 'covariance'")
    
    if choice == "correlation":
        return self.corr_
    elif choice == "covariance":
        return self.cov_
    
def get_candisc_coef(self,choice="absolute"):
    """
    Extract coefficients - CANDISC
    ------------------------------

    Description
    -----------
    Extract coefficients (canonical coefficients, coefficients of classification function) from Canonical Discriminant Analysis (CANDISC) outputs.

    Usage
    -----
    ```python
    >>> get_candisc_coef(self,choice=("absolute","score"))
    ```
    
    Parameters
    ----------
    self : an object of class CANDISC

    choice : the element to subset from the output. Allowed values are :
        * "absolute" for canonical coefficients
        * "score" for coefficients of classification function

    Returns
    -------
    pandas dataframe containing coefficients

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import CANDISC, get_candisc_coef
    >>> res_candisc = CANDISC(n_components=2,target=["species"],priors="prop",parallelize=False)
    >>> res_candisc.fit(iris)
    >>> # results for canonical coefficients
    >>> cancoef = get_candisc_coef(res_candisc,choice="absolute") 
    >>> # results for classification coefficients
    >>> classcoef = get_candisc_coef(res_candisc,choice="score") 
    ```
    """
    if self.model_ != "candisc":
        raise TypeError("'self' must be an object of class CANDISC")
    
    if choice == "absolute":
        return pd.concat((self.coef_,self.intercept_.to_frame().T),axis=0)
    elif choice == "score":
        return pd.concat((self.score_coef_,self.score_intercept_),axis=0)
    
def get_candisc(self,choice = "ind"):
    """
    Extract the results - CANDISC
    -----------------------------

    Description
    -----------
    Extract results (individuals coordinates, canonical correlation, covariance, canonical coefficients, coefficients of classification function) from Canonical Discriminant Analysis (CANDISC) outputs.

    Usage
    -----
    ```python
    >>> get_candisc(self,choice = ("ind","correlation","covariance","absolute","score"))
    ```

    Parameters
    ----------
    `self` : an object of class CANDISC

    choice : the element to subset from the output. Allowed values are : 
        * "ind" for individuals coordinates
        * "correlation" for canonical correlation
        * "covariance" for covariance
        * "absolute" for canonical coefficients
        * "score" for coefficients of classification function.

    Returns
    -------
    a dictionary or a pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import CANDISC, get_candisc
    >>> res_candisc = CANDISC(n_components=2,target=["species"],priors="prop",parallelize=False)
    >>> res_candisc.fit(iris)
    >>> # Results for individuals
    >>> ind = get_candisc(res_candisc, choice = "ind")
    >>> # Results for variables - Canonical correlation
    >>> cancorr = get_candisc(res_candisc, choice = "correlation")
    >>> # Results for variables - covariance
    >>> covar = get_candisc(res_candisc, choice = "covariance")
    >>> # Canonical coefficients
    >>> cancoef = get_candisc(res_candisc, choice = "absolute")
    >>> # Coefficients of classification function
    >>> classcoef = get_candisc(res_candisc, choice = "score")
    ```
    """
    if self.model_ != "candisc":
        raise TypeError("'self' must be an object of class CANDISC")
    
    if choice not in ["ind","correlation","covariance","absolute","score"]:
        raise ValueError("'choice' should be one of 'ind', 'correlation', 'covariance', 'absolute', 'score'")

    if choice == "ind":
        return get_candisc_ind(self)
    elif choice in ["correlation","covariance"]:
        return get_candisc_var(self,choice=choice)
    elif choice in ["absolute","score"]:
        return get_candisc_coef(self,choice=choice)

def summaryCANDISC(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Canonical Discriminant Analysis model
    -----------------------------------------------------------

    Description
    -----------
    Printing summaries of canonical discriminant analysis objects.

    Usage
    -----
    ```python
    >>> summaryCANDISC(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    self :   an object of class CANDISC

    digits :   int, default=3. Number of decimal printed

    nb_element :   int, default = 10. Number of element written

    ncp :   int, default = 3. Number of componennts

    to_markdown :   print summaries in markdown-friendly format.

    tablefmt :   table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs :   these parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import CANDISC, summaryCANDISC
    >>> res_candisc = CANDISC(n_components=2,target=["species"],priors="prop",parallelize=False)
    >>> res_candisc.fit(iris)
    >>> summaryCANDISC(res_candisc)
    ```
    """

    if self.model_ != "candisc":
        raise ValueError("'self' must be an object of class CANDISC")

    # Define number of components
    ncp = min(ncp,self.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])
    
    ind = get_candisc(self,choice="ind")
    vcorr = get_candisc(self,choice="correlation")
    coef = get_candisc_coef(self,choice="absolute").round(decimals=digits)
    score_coef = get_candisc_coef(self,choice="score").round(decimals=digits)

    # Partial Principal Components Analysis Results
    print("                     Canonical Discriminant Analysis - Results                     \n")

    print("\nSummary Information\n")
    summary = self.summary_information_
    if to_markdown:
        print(summary.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(summary)
    
    print("\nClass Level information\n")
    class_level_infos = self.statistics_["information"]
    if to_markdown:
        print(class_level_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(class_level_infos)

    # Add eigenvalues informations
    print("\nImportance of components")
    eig = self.eig_.T.round(decimals=digits)
    eig.index = ["Variance","Difference","% of var.","Cumulative % of var."]
    if to_markdown:
        print(eig.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(eig)
    
    print("\nTest of H0: The canonical correlations in the current row and all that follow are zero\n")
    lrt_test = self.statistics_["likelihood_test"].round(decimals=digits)
    if to_markdown:
        print(lrt_test.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(lrt_test)
    
    print("\nGroup means:\n")
    gmean = self.classes_["mean"]
    gmean.index.name = None
    gmean = gmean.T.round(decimals=digits)
    if to_markdown:
        print(gmean.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(gmean)
    
    print("\nCoefficients of canonical discriminants:\n")
    if to_markdown:
        print(coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(coef)
    
    print("\nClassification functions coefficients:\n")
    if to_markdown:
        print(score_coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(score_coef)

    # Add individuals informations
    if self.ind_["coord"].shape[0]>nb_element:
        print(f"\nIndividuals (the {nb_element} first)\n")
    else:
         print("\nIndividuals\n")
    ind_infos = ind["coord"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)
    
    # Add variables informations
    if self.corr_["total"].shape[0]>nb_element:
        print(f"\nCorrelations between Canonical and Original Variables (the {nb_element} first)\n")
    else:
         print("\nCorrelations between Canonical and Original Variables\n")
    var_infos = pd.DataFrame().astype("float")
    for i in np.arange(0,ncp,1):
        tcorr = vcorr["total"].iloc[:,i]
        tcorr.name ="total."+str(i+1)
        bcorr = vcorr["between"].iloc[:,i]
        bcorr.name ="between."+str(i+1)
        wcorr = vcorr["within"].iloc[:,i]
        wcorr.name ="within."+str(i+1)
        var_infos = pd.concat([var_infos,tcorr,bcorr,wcorr],axis=1)
    var_infos = var_infos.round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    print("\nClass Means on Canonical Variables\n")
    gcoord = self.classes_["coord"].round(decimals=digits)
    if to_markdown:
        print(gcoord.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(gcoord)