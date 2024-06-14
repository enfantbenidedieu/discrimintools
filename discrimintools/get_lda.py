# -*- coding: utf-8 -*-
import pandas as pd

def get_lda_ind(self):
    """
    Extract the results for individuals - LDA
    -----------------------------------------

    Description
    -----------
    Extract the results for active individuals from Linear Discriminant Analysis (LDA) outputs.

    Usage
    -----
    ```python
    >>> get_lda_ind(self)
    ```

    Parameters
    ----------
    `self` : an object of class LDA

    Returns
    -------
    dictionary of dataframes containing all the results for the active individuals including:

    `scores` : scores for the individuals

    `generalied_dist2` : generalized square distance

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import LDA, get_lda_ind
    >>> res_lda = LDA(target=["species"],priors="prop")
    >>> res_lda.fit(iris)
    >>> # Results for individuals
    >>> ind = get_lda_ind(res_lda)
    ```
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    return self.ind_

def get_lda_var(self):
    """
    Extract the results for variables - LDA
    ---------------------------------------

    Description
    -----------
    Extract the results (covariance) for variables from Linear Discriminant Analysis (LDA) outputs.

    Usage
    -----
    ```python
    >>> get_lda_var(self)
    ```

    Parameters
    ----------
    `self` : an object of class LDA

    Returns
    -------
    dictionary of dataframes containings all the results for the variables

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # Load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import LDA, get_lda_var
    >>> res_lda = LDA(target=["species"],priors="prop")
    >>> res_lda.fit(iris)
    >>> # covariance
    >>> covar = get_lda_var(res_lda) 
    ```
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    return self.cov_
    
def get_lda_coef(self):
    """
    Extract coefficients - LDA
    --------------------------

    Description
    -----------
    Extract coefficients of classification function from Linear Discriminant Analysis (LDA) outputs.

    Usage
    -----
    ```python
    >>> get_lda_coef(self)
    ```

    Parameters
    ----------
    `self` : an object of class LDA

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
    >>> from discrimintools import LDA, get_lda_coef
    >>> res_lda = LDA(target=["species"],priors="prop")
    >>> res_lda.fit(iris)
    >>> # results for classification coefficients
    >>> classcoef = get_lda_coef(res_lda) 
    ```
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    return pd.concat((self.coef_,self.intercept_),axis=0)
     
def get_lda(self,choice = "ind"):
    """
    Extract the results - LDA
    -------------------------

    Description
    -----------
    Extract results (individuals, covariance, coefficients of classification function) from Linear Discriminant Analysis (LDA) outputs.

    Usage
    -----
    ```python
    >>> get_lda(self,choice=("ind","cov","coef"))
    ```

    Parameters
    ----------
    `self` : an object of class LDA

    `choice` : the element to subset from the output. Allowed values are :
        * "ind" for individuals
        * "cov" for covariance
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
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import LDA, get_lda
    >>> res_lda = LDA(target=["species"],priors="prop")
    >>> res_lda.fit(iris)
    >>> # Results for individuals
    >>> ind = get_lda(res_lda, choice = "ind")
    >>> # Results for variables - covariance
    >>> covar = get_lda(res_lda, choice = "cov")
    >>> # Coefficients of classification function
    >>> classcoef = get_lda(res_lda, choice = "coef")
    ```
    """
    if self.model_ != "lda":
        raise TypeError("'self' must be an object of class LDA")
    
    if choice not in ["ind","cov","coef"]:
        raise ValueError("'choice' should be one of 'ind', 'cov', 'coef'")

    if choice == "ind":
        return get_lda_ind(self)
    elif choice == "cov":
        return get_lda_var(self)
    elif choice == "coef":
        return get_lda_coef(self)

def summaryLDA(self,digits=3,nb_element=10,to_markdown=False,tablefmt = "pipe",**kwargs):
    """
    Printing summaries of Linear Discriminant Analysis model
    --------------------------------------------------------

    Description
    -----------
    Printing summaries of linear discriminant analysis objects.

    Usage
    -----
    ```python
    >>> summaryLDA(self,digits=3,nb_element=10,to_markdown=False,tablefmt = "pipe",**kwargs)
    ```

    Parameters
    ----------
    `self` : an object of class LDA

    `digits` : int, default=3. Number of decimal printed

    `nb_element` : int, default = 10. Number of element

    `to_markdown` : Print DataFrame in Markdown-friendly format.

    `tablefmt` : Table format. For more about tablefmt, see : https://pypi.org/project/tabulate/

    `**kwargs` : These parameters will be passed to tabulate.

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> # load iris dataset
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import LDA, summaryLDA
    >>> res_lda = LDA(target=["species"],priors="prop")
    >>> res_lda.fit(iris)
    >>> summaryLDA(res_lda)
    ```
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