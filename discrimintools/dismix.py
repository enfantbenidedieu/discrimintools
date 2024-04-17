# -*- coding: utf-8 -*-

import scipy.stats as st
import numpy as np
import pandas as pd
from functools import reduce, partial
from scipy.spatial.distance import pdist,squareform
from statsmodels.multivariate.manova import MANOVA
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

##################################################################################################
#           Linear Discriminant Analysis with both Continuous and Categorical variables (DISMIX)
###################################################################################################
class DISMIX(BaseEstimator,TransformerMixin):
    """Discriminant Analysis under Continuous and Categorical variables (DISMIX)

    Performs linear discriminant analysis with both continuous and catogericals variables

    Parameters:
    -----------
    n_components
    target :
    quanti_features_labels :
    quali_features_labels:
    row_labels :
    prioirs:
    grid_search : 

    
    
    """
    def __init__(self,
                 n_components = None,
                 target=list[str],
                 quanti_features_labels=list[str],
                 quali_features_labels = list[str],
                 row_labels = list[str],
                 priors=None,
                 parallelize=False):
        self.n_components = n_components
        self.target = target
        self.quanti_features_labels = quanti_features_labels
        self.quali_features_labels = quali_features_labels
        self.row_labels = row_labels
        self.priors = priors
        self.parallelize = parallelize

    def fit(self,X,y=None):
        """Fit the Linear Discriminant Analysis with categories variables

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

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Set parallelize
        if self.parallelize:
            self.n_workers_ = -1
        else:
            self.n_workers_ = 1

        # Save data
        self.data_ = X
        
        self.target_ = self.target
        if self.target_ is None:
            raise ValueError("Error :'target' must be assigned.")
        
        self.quanti_features_labels_ = self.quanti_features_labels
        if self.quanti_features_labels_ is None:
            raise ValueError("Error :'quanti_features_labels' must be assigned.")
        
        self.quali_features_labels_ = self.quali_features_labels
        if self.quali_features_labels_ is None:
            raise ValueError("Error :'quali_features_labels' must be assigned.")
        
        self._compute_stats(X=X)
        
        return self
    
    def _compute_stats(self,X):
        """
        
        
        """

        # 

        # Continuous variables
        x = X.drop(columns=self.target_)
        # Qualitative variables - target
        y = X[self.target_]

        # categories
        self.classes_ = np.unique(y)

        # Number of observations
        self.n_samples_, self.n_features_ = x.shape

        # Number of groups
        self.n_classes_ = len(self.classes_)

        # Set row labels
        self.row_labels_ = self.row_labels
        if self.row_labels_ is None:
            self.row_labels = ["row."+str(i+1) for i in np.arange(0,self.n_samples_)]

        # Analyse Factoreielle sur données mixtes (FAMD)
        famd = FAMD(normalize=True,
                    n_components=self.n_components,
                    row_labels=self.row_labels_,
                    quali_labels=self.quali_features_labels_,
                    quanti_labels=self.quanti_features_labels_,
                    quali_sup_labels=None,
                    quanti_sup_labels=None,
                    row_sup_labels=None,
                    parallelize=self.parallelize).fit(x)
        
        # Extraction des informations sur les individus
        row = get_famd_ind(famd)

        # Coordonnées issues de l'Analyse en Composantes Principales (AFDM est sur ACP sur des données transformées)
        var_mod_coord = pd.DataFrame(famd.var_mod_coord_,index=famd.col_labels_+list(famd.mod_labels_),columns=famd.dim_index_)

        # Coefficients de projections sur les modalités des variables qualitatives
        fproj1 = mapply(var_mod_coord.loc[famd.mod_labels_,:],lambda x : x/(len(self.quali_features_labels_)*np.sqrt(famd.eig_[0])),axis=1,progressbar=False,n_workers=self.n_workers_)

        # Coefficients des fonctions de projection sur les variables quantitatives
        fproj2 = mapply(var_mod_coord.loc[famd.col_labels_,:],lambda x : x/(len(self.quanti_features_labels_)*np.sqrt(famd.eig_[0])),axis=1,progressbar=False,n_workers=self.n_workers_)

        # Concaténation des fonction des projection
        fproj = pd.concat([fproj2,fproj1],axis=0)

        # Données pour l'Analyse Discriminante Linéaire (LDA)
        row_coord = row["coord"]
        row_coord.columns = list(["Z"+str(x+1) for x in np.arange(0,famd.n_components_)])
        new_X = pd.concat([y,row_coord],axis=1)

        # Analyse Discriminante Linéaire
        lda = LDA(target=self.target_,
                  distribution="homoscedastik",
                  features_labels=row_coord.columns,
                  row_labels=self.row_labels_,
                  priors=self.priors,
                  parallelize=self.parallelize).fit(new_X)
        
        # LDA coefficients and intercepts
        lda_coef = lda.coef_
        lda_intercept = lda.intercept_ 

        # Coefficient du MIXDISC
        coef = pd.DataFrame(np.dot(fproj,lda_coef),columns=lda_coef.columns,index=fproj.index)
        
        # Sortie de l'ACM
        self.projection_function_ = fproj
        self.n_components_ = famd.n_components_

        # Sortie de l'ADL
        self.lda_coef_ = lda_coef
        self.lda_intercept_ = lda_intercept
        self.lda_features_labels_ = list(["Z"+str(x+1) for x in np.arange(0,famd.n_components_)])
        
        # Informations du MIXDISC
        self.statistical_evaluation_ = lda.statistical_evaluation_
        self.coef_ = coef
        self.intercept_ = lda_intercept

        # Stockage des deux modèles
        self.famd_model_ = famd
        self.lda_model_ = lda

        self.model_ = "dismix"
    
    def fit_transform(self,X):
        """
        Fit to data, then transform it.

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

        return self.famd_model_.row_coord_
    
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

        coord = self.famd_model_.transform(X)
        coord = pd.DataFrame(coord[:,:self.n_components_],index=X.index,columns=self.lda_features_labels_)

        return coord
    
    def predict(self,X):
        """Predict class labels for samples in X

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
        """Estimate probability

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
        Return the mean accuracy on the given test data and labels.

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