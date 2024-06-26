# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from scipy.spatial.distance import pdist,squareform
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

from scientisttools import CA


##########################################################################################
#           Discriminant Corrrespondence Analysis (DISCA)
##########################################################################################
class DISCA(BaseEstimator,TransformerMixin):
    """
    Discriminant Correspondence Analysis (DISCA)
    --------------------------------------------
    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Description
    -----------
    Performs Discriminant Correspondence Analysis model.

    Usage
    -----
    ```python
    >>> DISCA(n_components = 2, target = None, features = None, priors = None, parallelize = False)
    ```

    Parameters
    ----------
    `n_components` : number of dimensions kept in the results (by default 2)

    `target` : list of string with length 1 specifying the values of the classification variable define the groups for analysis.

    `features` : list of qualitatives variables to be included in the analysis. The default is all categoricals variables in dataset

    `priors` : The priors statement specifying the prior probabilities of group membership.
        * "equal" to set the priors probabilities equal,
        * "proportional" or "prop" to set the priors probabilities proportional to the sample sizes
        * pandas series which specify the prior probability for each level of the classification variable.

    `parallelize` : boolean, default = False. If model should be parallelize
        * If `True` : parallelize using mapply (see https://mapply.readthedocs.io/en/stable/README.html#installation)
        * If `False` : parallelize using apply

    Attributes
    ----------
    `call_` : dictionary with some statistics

    `ind_` : dictionary of pandas dataframe containing all the results for the active individuals (coordinates)

    `var_` : dictionary of pandas dataframe containing all the results for the active variables (coordinates, correlation between variables and axes, square cosine, contributions)

    `statistics_` : dictionary containing some statistics results :
        * `chi2` : pandas dataframe containing the chi-squared test between features and target
        * `categories` : pandas dataframe containing the distribution in count for features
        * `information` : pandas dataframe containing the class level information (frequence, proportion, priors)

    `classes_` : dictionary containing the classes information:
        * `coord` : pandas dataframe containing the group coordinates
        * `cos2` : pandas dataframe containing the group cos2
        * `contrib` : pandas dataframe containing the group contributions
        * `infos` : pandas dataframe containing the group additional informations (margin, square distance to origin & inertia)
        * `classes` : name of categories
        * `dist2` : pandas dataframe containing the square distance to origin for groups
        * `dist` : pandas dataframe containing the square distance between each group

    `anova_` : dictionary containing :
        * `eta2` : pandas dataframe containing the square correlation ratio
        * `canonical_eta2` : pandas dataframe containing the canonical square correlation ratio

    `factor_model_` : correspondence analysis (CA) model

    `coef_` : pandas dataframe containing discriminant correspondence analysis coefficients

    `model_` : string specifying the model fitted = 'disca'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    Abdi, H., and Williams, L.J. (2010). Principal component analysis. Wiley Interdisciplinary Reviews: Computational Statistics, 2, 433-459.
    
    Abdi, H. and Williams, L.J. (2010). Correspondence analysis. In N.J. Salkind, D.M., Dougherty, & B. Frey (Eds.): Encyclopedia of Research Design. Thousand Oaks (CA): Sage. pp. 267-278.
    
    Abdi, H. (2007). Singular Value Decomposition (SVD) and Generalized Singular Value Decomposition (GSVD). In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics.Thousand Oaks (CA): Sage. pp. 907-912.
    
    Abdi, H. (2007). Discriminant correspondence analysis. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics. Thousand Oaks (CA): Sage. pp. 270-275.

    Ricco Rakotomalala (2020), Pratique de l'analyse discriminante linéaire, Version 1.0, 2020

    Links
    -----
    https://bookdown.org/teddyswiebold/multivariate_statistical_analysis_using_r/discriminant-correspondence-analysis.html

    https://search.r-project.org/CRAN/refmans/TExPosition/html/tepDICA.html

    http://pbil.univ-lyon1.fr/ADE-4/ade4-html/discrimin.coa.html

    https://rdrr.io/cran/ade4/man/discrimin.coa.html

    https://stat.ethz.ch/pipermail/r-help/2010-December/263170.html

    https://www.sciencedirect.com/science/article/pii/S259026012200011X 

    See also
    --------
    get_disca_ind, get_disca_var, get_disca_classes, get_disca_coef, get_disca, summaryDISCA, fviz_disca_ind, fviz_disca_mod

    Examples
    --------
    ```python
    >>> # load canines dataset
    >>> from discrimintools.datasets import load_canines
    >>> canines = load_canines()
    >>> from discrimintools import DISCA
    >>> res_disca = DISCA(n_components=2,target=["Fonction"],priors = "prop")
    >>> res_disca.fit(canines)
    ```
    """
    def __init__(self,
                 n_components = 2,
                 target = None,
                 features = None,
                 priors = None,
                 parallelize = False):
        self.n_components = n_components
        self.target = target
        self.features = features
        self.priors = priors
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """
        Fit the Discriminant Correspondence Analysis model
        --------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_features+1)
            Training Data.
        
        `y` : None.
            y is ignored.
        
        Returns
        -------
        `self` : object
            Fitted estimator
        """ 
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()     
        
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Check if target is None
        if self.target is None:
            raise ValueError("'target' must be assigned")
        elif not isinstance(self.target,list):
            raise TypeError("'target' must be a list")
        elif len(self.target)>1:
            raise ValueError("'target' must be a list of length one")

        ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
         # Save data
        Xtot = X.copy()
        
        #######################################################################################################################
        # Split Data into two : x and y
        y = X[self.target]
        x = X.drop(columns=self.target)

        # Set features labels/names
        if self.features is None:
            features = x.columns.tolist()
        elif not isinstance(self.features,list):
            raise ValueError("'features' must be a list of variable names")
        else:
            features = self.features
        
        ###### Select features
        x = x[features]
        
        # Redefine X
        X = pd.concat((x,y),axis=1)

        # Number of rows and continuous variables
        n_samples, n_features = x.shape

        ##### Category
        classes = np.unique(y).tolist()
        # Number of groups
        n_classes = len(classes)

         ################################################ Check if all columns are categoricals
        all_cat = all(pd.api.types.is_string_dtype(x[col]) for col in x.columns)
        if not all_cat:
            raise TypeError("All features must be categoricals")
        
        ################################################ Chi2 -test
        chi2_test = pd.DataFrame(columns=["statistic","ddl","pvalue"],index=x.columns).astype("float")
        for col in x.columns:
            tab = pd.crosstab(x[col],y[self.target[0]])
            chi2 = sp.stats.chi2_contingency(observed=tab,correction=False)
            chi2_test.loc[col,:] = [chi2[0], chi2[2], chi2[1]]
        chi2_test = chi2_test.sort_values(by=["pvalue"])
        chi2_test["ddl"] = chi2_test["ddl"].astype(int)
        statistics = {"chi2" : chi2_test}

        ######################################################################################
        # Tableau des indicatrices
        dummies = pd.concat((pd.get_dummies(x[col],prefix=col,prefix_sep="_",dtype=int) for col in x.columns),axis=1)

        ############################################################
        # Construction de la matrice M
        M = pd.concat([y,dummies],axis=1).groupby(self.target).sum()

        #########################################################################################
        #   Correspondence Analysis (CA)
        ######################################################################################
        # Correspondence Analysis
        global_ca = CA(n_components=self.n_components,parallelize=self.parallelize).fit(M)
        
        # Calcul des profils - Effectifs par classes (pour chaque descripteurs)
        n_l = M.sum(axis=0)
        # Profil marginal
        G = n_l/np.sum(n_l)

        # Tableau des profils - Matric des profils
        profils = mapply(M,lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)
        
        # Distance entre un groupe et l'origine
        dist2 = mapply(profils,lambda x : np.sum((x-G.values)**2/G.values),axis=1,progressbar=False,n_workers=n_workers).to_frame("dist2")

        # Distance entre les groupes - Matrice des distances par paires de classes
        dist = pd.DataFrame(squareform(pdist(profils,metric="seuclidean",V=G)**2),index=classes,columns=classes)

        ###########################################
        mod_stats = pd.concat([n_l,G],axis=1)
        mod_stats.columns = ["Frequence","Proportion"]
        statistics = {**statistics,**{"categories" : mod_stats}}
        
        # Number of element 
        n_k, p_k = y.value_counts(normalize=False), y.value_counts(normalize=True)

        # Initial prior - proportion of each element
        if self.priors is None:
            raise ValueError("'priors' must be assigned")

        if self.priors in ["proportional","prop"]:
            priors = p_k
        elif self.priors == "equal":
            priors = pd.Series([1/n_classes]*n_classes,index=classes)
        else:
            priors = pd.Series([x/self.priors.sum() for x in self.priors.values],index=self.priors.index)

        # Store some informations
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "target" : self.target[0],
                      "features" : features,
                      "n_features" : n_features,
                      "n_samples" : n_samples,
                      "n_classes" : n_classes,
                      "priors" : priors}
        
        #############################
        # Class level information
        class_level_information = pd.concat([n_k,p_k,priors],axis=1)
        class_level_information.columns = ["Frequency","Proportion","Prior Probability"]
        statistics["information"] = class_level_information
        self.statistics_ = statistics

        #######################################################
        # Coefficient des fonctions discriminantes canoniques
        coef = mapply(global_ca.col_["coord"],lambda col : col/(x.shape[1]*np.sqrt(global_ca.eig_.iloc[:global_ca.call_["n_components"],0])),
                      axis=1,progressbar=False,n_workers=n_workers)
        
        ####################################################################
        # Coordonnées des individus à partir du tableau des indicatrices
        row_coord = dummies.dot(coef)
        self.ind_ = {"coord" : row_coord}
        
        # Somme des carrés totales - totla sum of squared
        tss = mapply(row_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=n_workers).sum(axis=0)
        
        # Rapport de corrélation - Correlation ratio
        eta2 = ((x.shape[0]*global_ca.eig_.iloc[:,0])/tss)
        eta2.name = "eta2"

        ##################################################################################
        # Information sur les classes
        self.classes_ = global_ca.row_
        self.classes_ = {**self.classes_, **{"classes" : classes, "dist2" : dist2, "dist" : dist}}

        ##################################################################################
        # Informations sur les categories
        self.var_ = global_ca.col_

        #################################################################################
        # Correspondance Analysis
        self.factor_model_ = global_ca

        ########################### DISCA Coefficients
        # Score function
        self.coef_ = coef
        
        # Analysis of variance
        self.anova_ = {"eta2" : eta2,"canonical_eta2" : mapply(eta2,lambda x : np.sqrt(x),axis=0,progressbar=False,n_workers=n_workers)}

        self.model_ = "disca"
        
        return self
        
    def fit_transform(self,X,y=None):
        """
        Fit to data, then transform it
        ------------------------------

        Description
        -----------
        Fits transformer to `X` and returns a transformed version of `X`.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_features+1)
            Input samples.
        
        `y` : None.
            y is ignored.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples, n_components)
            Transformed dataframe.
        """
        self.fit(X)
        return self.ind_["coord"]
    
    def transform(self,X,y=None):
        """ 
        Apply the dimensionality reduction on X
        --------------------------------------- 
        
        Description
        -----------
        X is projected on the first axes previous extracted from a training set.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples_plus, n_features) or (n_samples_plus, n_features+1)
        
        `y` : None
            y is ignored.

        Returns
        -------
        `X_new` : pandas dataframe of shape (n_samples_sup, n_components)
            Coordinates of the projections of the supplementary
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
         ##### Chack if target in X columns
        if self.call_["target"] in X.columns.tolist():
            X = X.drop(columns=[self.call_["target"]])
        
        ####### Select features
        X = X[self.call_["features"]]

        #self._compute_row_sup_stats(X)
        n_rows = X.shape[0]
        n_cols = self.var_["coord"].shape[0]
        Y = np.zeros((n_rows,n_cols))
        for i in np.arange(0,n_rows,1):
            values = [X.columns[k] +"_"+str(X.iloc[i,k]) for k in np.arange(X.shape[1])]
            for j in np.arange(n_cols):
                if self.var_["coord"].index.tolist()[j] in values:
                    Y[i,j] = 1
        row_sup_dummies = pd.DataFrame(Y,columns=self.var_["coord"].index,index=X.index)
        return row_sup_dummies.dot(self.coef_)
    
    def decision_function(self,X):
        """
        Apply decision function to a pandas dataframe of samples
        --------------------------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_features)
            
        Returns
        -------
        `C` : pandas dataframe of shape (n_samples_,) or (n_samples_, n_classes)
            Decision function values related to each class, per sample.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        # Check if X is a pandas DataFrame
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Coordonnées des individus
        coord = self.transform(X)

        # Distance euclidiennes aux centres de classes
        scores = pd.concat((mapply(self.classes_["coord"].sub(coord.loc[i,:].values,axis="columns"),lambda x : np.sum(x**2),
                                   axis=1,progressbar=False,n_workers=n_workers).to_frame(i).rename_axis(None).T 
                            for i in coord.index),axis=0)
        return scores

    def predict_proba(self,X):
        """
        Estimate probability
        --------------------

        Parameters
        ----------
        `X` : pandas/polars dataFrame of shape (n_samples, n_features)
            Input data.
        
        Returns:
        --------
        `C` : pandas dataframe of shape (n_samples,n_classes)
            Estimated probabilities.
        """

        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        # Decision
        scores = self.decision_function(X)

        # Distance généralisée : scores - 2*log(p_k)
        DG = scores.sub((2*np.log(self.call_["priors"].to_frame(name="prior").T.loc[:,scores.columns].values)),axis="columns")
    
        # Probabilité d'appartenance - transformation 
        C = mapply(mapply(DG,lambda x : np.exp(-0.5*x),axis=0,progressbar=False,n_workers=n_workers),
                   lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)
        return C
    
    def predict(self,X):
        """
        Predict class labels for samples in X
        -------------------------------------

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_features)
            The pandas/polars dataframe for which we want to get the predictions.
        
        Returns
        -------
        `y_pred` : pandas series of shape (n_samples,)
            Pandas series containing the class labels for each sample
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        predict_proba = self.predict_proba(X)
        predict = np.unique(self.classes_["classes"])[np.argmax(predict_proba.values,axis=1)]
        return pd.Series(predict,index=X.index,name="prediction")
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels
        ----------------------------------------------------------

        Notes
        -----
        In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        `X` : pandas/polars dataframe of shape (n_samples, n_features)
            Test samples.

        `y` : pandas series/dataframe of shape (n_samples,) or (n_samples,1)
            True labels for `X`.

        `sample_weight` : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        `score` : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def pred_table(self):
        """
        Prediction table
        ----------------
        
        Notes
        -----
        pred_table[i,j] refers to the number of times “i” was observed and the model predicted “j”. Correct predictions are along the diagonal.
        """
        pred = self.predict(self.call_["X"])
        return pd.crosstab(self.call_["X"][self.call_["target"]],pred)