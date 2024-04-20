# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scientisttools.eta2 import eta2
import scipy.stats as st

# Row informations
def get_disca_ind(self):
    """
    Extract the results for individuals - DISCA
    -------------------------------------------

    Parameters
    ----------
    self : an object of class DISCA

    Returns
    -------
    a dictionary of dataframes containing all the results for the active individuals including:
    - coord : coordinates for the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.ind_

# Categories informations
def get_disca_var(self):
    """
    Extract the results for variables/categories - DISCA
    ----------------------------------------------------

    Parameters
    ----------
    self : an object of class DISCA

    Returns
    -------
    a dictionary of dataframes containing all the results for the active variables including:
    - coord : coordinates for the variables/categories

    - contrib : contributions for the variables/categories

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.var_

# Group informations
def get_disca_classes(self):
    """
    Extract the results for groups - DISCA
    --------------------------------------

    Parameters
    ----------
    self : an object of class DISCA

    Returns
    -------
    a dictionary of dataframes containing all the results for the groups including:
    - coord : coordinates for the individuals

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.classes_

def get_disca_coef(self):
    """
    Extract coefficients - DISCA
    ----------------------------

    Parameters
    ----------
    self : an object of class DISCA

    Returns
    -------
    a pandas dataframe containing coefficients

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    return self.coef_

# Disca extract informations
def get_disca(self,choice="ind"):
    """
    Extract the results - DISCA
    ---------------------------

    Parameters
    ----------
    self : an object of class DISCA

    choice :

    Returns
    -------
    a dictionary or a pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
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

    Parameters
    ----------
    self        :   an object of class DISCA

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    ncp         :   int, default = 3. Number of componennts

    to_markdown :   Print DataFrame in Markdown-friendly format.

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "disca":
        raise ValueError("'self' must be an object of class DISCA")

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
    
    print("\nCanonical coeffcients\n")
    if to_markdown:
        print(coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(coef)

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