# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_lda_ind(self):
    """
    Extract the results for individuals - LDA
    -----------------------------------------

    Parameters
    ----------
    self : an object of class LDA

    Returns
    -------
    a dictionary of dataframes containing all the results for the active individuals including:
    - scores : scores for the individuals

    - generalied_dist2 : generalized distance

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    return self.ind_

def get_lda_cov(self):
    """
    Extract the results for variables - LDA
    ---------------------------------------

    Parameters
    ----------
    self : an object of class LDA

    Returns
    -------
    a dictionary of dataframes containings all the results for the variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    return self.cov_
    
def get_lda_coef(self):
    """
    Extract coefficients - LDA
    --------------------------

    Parameters
    ----------
    self : an object of class LDA

    Returns
    -------
    a pandas dataframe containing coefficients

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    return pd.concat((self.coef_,self.intercept_),axis=0)
    
    
def get_lda(self,choice = "ind"):
    """
    Extract the results - LDA
    -------------------------

    Parameters
    ----------
    self : an object of class LDA

    choice :

    Returns
    -------
    a dictionary or a pandas dataframe

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    
    if choice not in ["ind","cov","coef"]:
        raise ValueError("'choice' should be one of 'ind', 'cov', 'coef'")

    if choice == "ind":
        return get_lda_ind(self)
    elif choice == "cov":
        return get_lda_cov(self)
    elif choice == "coef":
        return get_lda_coef(self)

def summaryLDA(self,digits=3,nb_element=10,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Linear Discriminant Analysis model
    -----------------------------------------------------------

    Parameters
    ----------
    self        :   an object of class LDA

    digits      :   int, default=3. Number of decimal printed

    nb_element  :   int, default = 10. Number of element

    to_markdown :   Print DataFrame in Markdown-friendly format.

    tablefmt    :   Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    **kwargs    :   These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """

    if self.model_ != "lda":
        raise ValueError("'self' must be an object of class LDA")

    # Define number of components
    nb_element = min(nb_element,self.call_["X"].shape[0])
    
    ind = get_lda_ind(self)
    coef = get_lda_coef(self).round(decimals=digits)

    # Partial Principal Components Analysis Results
    print("                     Linear Discriminant Analysis - Results                     \n")

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
    
    print("\nGroup means:\n")
    gmean = self.classes_["mean"]
    gmean.index.name = None
    gmean = gmean.T.round(decimals=digits)
    if to_markdown:
        print(gmean.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(gmean)
    
    print("\nCoefficients of linear discriminants:\n")
    if to_markdown:
        print(coef.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(coef)

    # Add individuals informations
    if self.ind_["scores"].shape[0]>nb_element:
        print(f"\nIndividuals (the {nb_element} first) scores\n")
    else:
         print("\nIndividuals scores\n")
    ind_infos = ind["scores"].iloc[:nb_element,:].round(decimals=digits)
    if to_markdown:
        print(ind_infos.to_markdown(tablefmt=tablefmt,**kwargs))
    else:
        print(ind_infos)