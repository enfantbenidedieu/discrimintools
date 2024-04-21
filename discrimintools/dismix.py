# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score
from scientisttools import FAMD

from .lda import LDA
from .eta2 import eta2
from .revaluate_cat_variable import revaluate_cat_variable

##################################################################################################
#           Linear Discriminant Analysis with both Continuous and Categorical variables (DISMIX)
###################################################################################################
class DISMIX(BaseEstimator,TransformerMixin):
    """
    Discriminant Analysis of Mixed Data (DISMIX)
    --------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs linear discriminant analysis with both continuous and catogericals variables

    Parameters:
    -----------
     n_components : number of dimensions kept in the results 

    target : The values of the classification variable define the groups for analysis.

    features : list of mixed variables to be included in the analysis

    priors : The priors statement specifies the prior probabilities of group membership.
                - "equal" to set the prior probabilities equal,
                - "proportional" or "prop" to set the prior probabilities proportional to the sample sizes
                - a pandas series which specify the prior probability for each level of the classification variable.
    
    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    call_ : a dictionary with some statistics

    coef_ : DataFrame of shape (n_features,n_classes_)

    intercept_ : DataFrame of shape (1, n_classes)

    lda_model_ : linear discriminant analysis model

    factor_model_ : factor analysis of mixed data model

    projection_function_ : projection function

    coef_ : pandas dataframe of shpz (n_categories, n_classes)

    intercept_ : pandas dataframe of shape (1, n_classes)

    model_ : string. The model fitted = 'dismix'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References:
    -----------
    Ricco Rakotomalala, Pratique de l'analyse discriminante linéaire, Version 1.0, 2020
    """
    def __init__(self,
                 n_components = None,
                 target=None,
                 features = None,
                 priors=None,
                 parallelize=False):
        self.n_components = n_components
        self.target = target
        self.features = features
        self.priors = priors
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """
        Fit the Linear Discriminant Analysis of Mixed Data model
        --------------------------------------------------------

        Parameters:
        -----------
        X : DataFrame of shape (n_samples, n_features+1)
            Training data
        
        y : None

        Returns:
        --------
        self : object
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
        
        # Check if target is None
        if self.target is None:
            raise ValueError("'target' must be assigned")
        elif not isinstance(self.target,list):
            raise ValueError("'target' must be a list")
        elif len(self.target)>1:
            raise ValueError("'target' must be a list of length one")
        
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
       ###############################################################################################################"
        # Drop level if ndim greater than 1 and reset columns name
        ###############################################################################################################
        if X.columns.nlevels > 1:
            X.columns = X.columns.droplevel()
        
        # Save data
        Xtot = X.copy()
        
        #######################################################################################################################
        # Split Data into two : X and y
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

        #########################################################################"
        # Categoricals variables
        cats = x.select_dtypes(include=["object","category"])
        # Continuous variables
        cont = x.drop(columns=cats.columns)

        # Check if no categoricals variables
        if cats.shape[0]==0:
            raise TypeError("No categoricals variables. Please use LDA instead")
        
        # Check if no continuous variables
        if cont.shape[0]==0:
            raise TypeError("No continuous variables. Please use DISQUAL")
        
        ################################################ Chi2 -test
        chi2_test = pd.DataFrame(columns=["statistic","ddl","pvalue"],index=cats.columns).astype("float")
        for col in cats.columns:
            tab = pd.crosstab(x[col],y[self.target[0]])
            chi2 = sp.stats.chi2_contingency(observed=tab,correction=False)
            chi2_test.loc[col,:] = [chi2[0], chi2[2], chi2[1]]
        chi2_test = chi2_test.sort_values(by=["pvalue"])
        chi2_test["ddl"] = chi2_test["ddl"].astype(int)

        ################################################ Correlation rato
        # Rapport de correlation - Correlation ration
        eta2_res = {}
        for col in cont.columns:
            eta2_res[col] = eta2(y,x[col])
        eta2_res = pd.DataFrame(eta2_res).T.sort_values(by=["pvalue"])
        self.statistics_ = {"chi2" : chi2_test, "Eta2" : eta2_res}

        #################################################################################################
        ####### Revaluate categoricals variables
        cats = revaluate_cat_variable(cats)
        x = pd.concat((cont,cats),axis=1)

        ##### Category
        classes = np.unique(y).tolist()
        # Number of groups
        n_classes = len(classes)

        # Number of rows and continuous variables
        n_samples, n_features = x.shape
        
        #############################################################""
        # Store some informations
        self.call_ = {"Xtot" : Xtot,
                      "X" : X,
                      "target" : self.target[0],
                      "features" : features,
                      "n_features" : n_features,
                      "n_samples" : n_samples,
                      "n_classes" : n_classes}

        ################################################################################
        # Factor Analysis of Mixed Data (FAMD)
        ##################################################################################
        global_famd = FAMD(n_components=self.n_components,parallelize=self.parallelize).fit(x)

        #######################################################################################################
        # Coefficients de projections sur les modalités des variables qualitatives
        fproj1 = mapply(global_famd.quali_var_["coord"],lambda x : x/(cats.shape[1]*np.sqrt(global_famd.eig_.iloc[:global_famd.call_["n_components"],0])),
                        axis=1,progressbar=False,n_workers=n_workers)

        # Coefficients des fonctions de projection sur les variables quantitatives
        fproj2 = mapply(global_famd.quanti_var_["coord"],lambda x : x/(cont.shape[1]*np.sqrt(global_famd.eig_.iloc[:global_famd.call_["n_components"],0])),
                        axis=1,progressbar=False,n_workers=n_workers)

        # Concaténation des fonction des projection
        fproj = pd.concat([fproj2,fproj1],axis=0)
        self.projection_function_ = fproj

        #########################################################################################################
        # Linear Discriminant Analysis (LDA)
        ########################################################################################################
        # Données pour l'Analyse Discriminante Linéaire (LDA)
        coord = global_famd.ind_["coord"].copy()
        coord.columns = ["Z"+str(x+1) for x in range(coord.shape[1])]
        data = pd.concat([y,coord],axis=1)

        # Linear Discriminant Analysis (LDA)
        lda = LDA(target=self.target,priors=self.priors).fit(data)

        ##################### DISMIX coeffcients and intercept
        self.coef_ = pd.DataFrame(np.dot(fproj,lda.coef_),index=fproj.index,columns=lda.coef_.columns)
        self.intercept_ = lda.intercept_

        # Stockage des deux modèles
        self.factor_model_ = global_famd
        self.lda_model_ = lda

        self.model_ = "dismix"

        return self
    
    def fit_transform(self,X):
        """
        Fit to data, then transform it
        ------------------------------

        Fits transformer to `X` and returns a transformed version of `X`.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features+1)
            Input samples.

        Returns
        -------
        X_new :  DataFrame of shape (n_samples, n_features_new)
            Transformed array.
        """
        self.fit(X)
        coord = self.factor_model_.ind_["coord"].copy()
        coord.columns = ["Z"+str(x+1) for x in range(coord.shape[1])]
        return coord
    
    def transform(self,X):
        """Project data to maximize class separation

        Parameters:
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        X_new : DataFrame of shape (n_samples, n_components_)
        
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

        ##### Chack if target in X columns
        if self.lda_model_.call_["target"] in X.columns.tolist():
            X = X.drop(columns=[self.lda_model_.call_["target"]])
        
        ####### Select features
        X = X[self.call_["features"]]
        
        # Categoricals variables
        cats = X.select_dtypes(include=["object","category"])
        cont = X.drop(columns=cats.columns)

        ####### Revaluate categoricals variables
        cats = revaluate_cat_variable(cats)
        X = pd.concat((cont,cats),axis=1)
        # Factor Analaysis of Mixed Data (FAMD)
        coord = self.factor_model_.transform(X).copy()
        coord.columns = ["Z"+str(x+1) for x in range(coord.shape[1])]
        return coord
    
    def predict(self,X):
        """
        Predict class labels for samples in X
        -------------------------------------

        Parameters:
        -----------
        X : DataFrame of shape (n_samples, n_features)
            The dataframe for which we want to get the predictions
        
        Returns:
        --------
        y_pred : DtaFrame of shape (n_samples, 1)
            DataFrame containing the class labels for each sample.
        
        """
        return self.lda_model_.predict(self.transform(X))
    
    def predict_proba(self,X):
        """
        Estimate probability
        --------------------

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data
        
        Returns:
        -------
        C : DataFrame of shape (n_samples, n_classes)
            Estimate probabilities
        
        """
        return self.lda_model_.predict_proba(self.transform(X))
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels
        ----------------------------------------------------------

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
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
        pred = self.predict(self.factor_model_.call_["X"])
        return pd.crosstab(self.lda_model_.call_["X"][self.lda_model_.call_["target"]],pred)