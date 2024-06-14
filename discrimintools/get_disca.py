# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def get_disca_ind(self):
    """
    Extract the results for individuals - DISCA
    -------------------------------------------

    Description
    -----------
    Extract the results (factor coordinates) for active individuals from Discriminant Correspondence Analysis (DISCA) outputs.

    Usage
    -----
    ````python
    >>> get_disca_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class DISCA

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
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA, get_disca_ind
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    >>> ind = get_disca_ind(res_disca)
    ```
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.ind_

def get_disca_var(self):
    """
    Extract the results for variables/categories - DISCA
    ----------------------------------------------------

    Description
    -----------
    Extract the results (factor coordinates, contributions, square cosinus) for variables/categories from Discriminant Correspondence Analysis (DISCA) outputs.

    Usage
    -----
    ```python
    >>> get_disca_var(self)
    ```

    Parameters
    ----------
    `self` : an object of class DISCA

    Returns
    -------
    dictionary of dataframes containing all the results for the active variables/categories including:

    `coord` : factor coordinates for the variables/categories

    `contrib` : contributions for the variables/categories

    `cos2` : square cosinus for variables/categories

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA, get_disca_var
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    >>> # Results for variables/categories
    >>> var = get_disca_var(res_disca)
    ```
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.var_

def get_disca_classes(self):
    """
    Extract the results for groups - DISCA
    --------------------------------------

    Description
    -----------
    Extract the results (factor coordinates, contributions, square cosinus) for groups/classes from Discriminant Correspondence Analysis (DISCA) outputs.

    Usage
    -----
    ```python
    >>> get_disca_classes(self)
    ```

    Parameters
    ----------
    `self` : an object of class DISCA

    Returns
    -------
    dictionary of dataframes containing all the results for the groups/classes including:

    `coord` : factor coordinates for the groups/classes

    `contrib` : contributions for the groups/classes

    `cos2` : square cosinus for the groups/classes

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA, get_disca_classes
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    >>> # Results for classes
    >>> classes = get_disca_classes(res_disca)
    ```
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.classes_

def get_disca_coef(self):
    """
    Extract coefficients - DISCA
    ----------------------------

    Description
    -----------
    Extract coefficients of classification function from Discriminant Correspondence Analysis (DISCA) outputs.

    Usage
    -----
    ```python
    >>> get_disca_coef(self)
    ```

    Parameters
    ----------
    `self` : an object of class DISCA

    Returns
    -------
    pandas dataframe containing coefficients

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA, get_disca_coef
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    >>> # Coefficients of classification function
    >>> classcoef = get_disca_coef(res_disca)
    ```
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.coef_

def get_disca(self,choice="ind"):
    """
    Extract the results - DISCA
    ---------------------------

    Description
    -----------
    Extract the results (individuals, variables, classes, coefficients) from Discriminant Correspondence Analysis (DISCA) outputs.

    Usage
    ------
    ```python
    >>> get_disca(self,choice=("ind","var","classes","coef"))
    ```

    Parameters
    ----------
    `self` : an object of class DISCA

    `choice` : the element to subset fro the output. Allowed values are :
        * "ind" for individuals
        * "var" for variables
        * "classes" for classes/groups
        * "coef" for coefficients of classification function

    Returns
    -------
    dictionary or pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA, get_disca
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    >>> # Results for individuals
    >>> ind = get_disca(res_disca, choice = "ind")
    >>> # Results for variables
    >>> var = get_disca(res_disca, choice = "var")
    >>> # Results for groups/classes
    >>> classes = get_disca(res_disca, choice = "classes")
    >>> # Coefficients of classification function
    >>> coef = get_disca(res_disca, choice = "coef")
    ```
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    
    if choice not in ["ind","var","classes","coef"]:
        raise ValueError("'choice' should be one of 'ind', 'var', 'classes', 'coef'")
    
    if choice == "ind":
        return get_disca_ind(self)
    elif choice == "var":
        return get_disca_var(self)
    elif choice == "classes":
        return get_disca_classes(self)
    elif choice == "coef":
        return get_disca_coef(self)

def summaryDISCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Discriminant Correspondence Analysis model
    ----------------------------------------------------------------

    Description
    -----------
    Printing summaries of discriminant correspondence analysis objects.

    Usage
    -----
    ```python
    >>> summaryDISCA(self,digits=3,nb_element=10,ncp=3,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class DISCA

    `digits` : int, default=3. Number of decimal printed

    `nb_element` :   int, default = 10. Number of element

    `ncp` :   int, default = 3. Number of componennts

    `to_markdown` : Print DataFrame in Markdown-friendly format.

    `tablefmt` : Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    `**kwargs` : These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA, summaryDISCA
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    >>> summaryDISCA(res_disca)
    ```
    """
    # Check if self is an object of class DISCA
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")

    # Define number of components
    ncp = min(ncp,self.factor_model_.call_["n_components"])
    nb_element = min(nb_element,self.call_["X"].shape[0])
    
    ind = get_disca(self,choice="ind")
    coef = self.coef_.round(decimals=digits)

    # Partial Principal Components Analysis Results
    print("                     Discriminant Correspondence Analysis - Results                     \n")

    print("\nClass Level information\n")
    class_level_infos = self.statistics_["information"]
    if to_markdown:
        print(class_level_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(class_level_infos)
    
    if coef.shape[0] > nb_element:
        print(f"\nCanonical coefficients (the {nb_element} first)\n")
    else:
        print("\nCanonical coeffcients\n")
    if to_markdown:
        print(coef.iloc[:nb_element,:].to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(coef.iloc[:nb_element,:])

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
    if self.var_["coord"].shape[0]>nb_element:
        print(f"\nCategories (the {nb_element} first)\n")
    else:
         print("\nCategories\n")
    var = self.var_
    var_infos = pd.DataFrame()
    for i in np.arange(0,ncp,1):
        var_coord = var["coord"].iloc[:,i]
        var_cos2 = var["cos2"].iloc[:,i]
        var_cos2.name = "cos2"
        var_ctr = var["contrib"].iloc[:,i]
        var_ctr.name = "ctr"
        var_infos = pd.concat([var_infos,var_coord,var_ctr,var_cos2],axis=1)
    var_infos = var_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(var_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(var_infos)
    
    # Add classes informations
    if self.classes_["coord"].shape[0]>nb_element:
        print(f"\nGroups (the {nb_element} first)\n")
    else:
         print("\nGroups\n")
    classes = self.classes_
    classes_infos = pd.DataFrame()
    for i in np.arange(0,ncp,1):
        classes_coord = classes["coord"].iloc[:,i]
        classes_cos2 = classes["cos2"].iloc[:,i]
        classes_cos2.name = "cos2"
        classes_ctr = classes["contrib"].iloc[:,i]
        classes_ctr.name = "ctr"
        classes_infos = pd.concat([classes_infos,classes_coord,classes_ctr,classes_cos2],axis=1)
    classes_infos = classes_infos.iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(classes_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(classes_infos)