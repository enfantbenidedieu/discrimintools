# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import polars as pl

from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

from scientisttools import PCA
from .lda import LDA

class PCADA(BaseEstimator,TransformerMixin):
    """
    Principal Components Analysis - Discriminant Analysis (PCADA)
    -------------------------------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs principal components analysis - discriminant analysis

    Parameters:
    ----------
    n_components : number of dimensions kept in the results 

    target : The values of the classification variable define the groups for analysis.

    features : list of quantitative variables to be included in the analysis. The default is all numeric variables in dataset

    priors : The priors statement specifies the prior probabilities of group membership.
                - "equal" to set the prior probabilities equal,
                - "proportional" or "prop" to set the prior probabilities proportional to the sample sizes
                - a pandas series which specify the prior probability for each level of the classification variable.
    
    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Returns:
    -------
    coef_ : DataFrame of shape (n_features,n_classes_)

    intercept_ : DataFrame of shape (1, n_classes)

    lda_model_ : linear discriminant analysis model

    factor_model_ : principal components analysis model

    projection_function_ : projection function

    coef_ : pandas dataframe of shpz (n_categories, n_classes)

    intercept_ : pandas dataframe of shape (1, n_classes)

    model_ : string. The model fitted = 'disqual'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    
    References:
    -----------
    Ricco Rakotomalala, Pratique de l'analyse discriminante linéaire, Version 1.0, 2020
    """
    def __init__(self,
                 n_components = None,
                 target = None,
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
        Fit the Linear Discriminant Analysis with categories variables model
        --------------------------------------------------------------------

        Parameters:
        -----------
        X : pandas/polars DataFrame of shape (n_samples, n_features+1)
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

        ################################################ Check if all columns are numerics
        # Check if all columns are numerics
        all_num = all(pd.api.types.is_numeric_dtype(x[c]) for c in x.columns.tolist())
        if not all_num:
            raise TypeError("All features must be numeric")

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
        
        ##################################################################################################################
        # Principal Components Analysis (PCA)
        ###################################################################################################################
        global_pca = PCA(standardize=True,n_components=self.n_components,parallelize=self.parallelize).fit(x)

        # Fonction de projection - Coefficient de projections
        fproj = mapply(global_pca.var_["coord"],lambda col : col/(x.shape[1]*np.sqrt(global_pca.eig_.iloc[:global_pca.call_["n_components"],0])),
                       axis=1,progressbar=False,n_workers=n_workers)
        self.projection_function_ = fproj

        ################################################################################################
        # Linear Discriminant Analysis (LDA)
        ##################################################################################################
        ###### Data to use in LDA
        coord = global_pca.ind_["coord"].copy()
        coord.columns = ["Z"+str(x+1) for x in range(coord.shape[1])]
        data = pd.concat([y,coord],axis=1)

        # Linear Discriminant Analysis model
        lda = LDA(target=self.target,priors=self.priors).fit(data)
        
        ##################### DISQUAL coeffcients and intercept
        self.coef_ = pd.DataFrame(np.dot(fproj,lda.coef_),index=fproj.index,columns=lda.coef_.columns)
        self.intercept_ = lda.intercept_

        # Stockage des deux modèles
        self.factor_model_ = global_pca
        self.lda_model_ = lda

        self.model_ = "disqual"
    
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
        """
        Project data to maximize class separation
        -----------------------------------------

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