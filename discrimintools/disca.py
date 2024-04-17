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


##########################################################################################
#           Discriminant Corrrespondence Analysis (DISCA)
##########################################################################################

# https://bookdown.org/teddyswiebold/multivariate_statistical_analysis_using_r/discriminant-correspondence-analysis.html
# https://search.r-project.org/CRAN/refmans/TExPosition/html/tepDICA.html
# http://pbil.univ-lyon1.fr/ADE-4/ade4-html/discrimin.coa.html
# https://rdrr.io/cran/ade4/man/discrimin.coa.html
# https://stat.ethz.ch/pipermail/r-help/2010-December/263170.html
# https://www.sciencedirect.com/science/article/pii/S259026012200011X

class DISCA(BaseEstimator,TransformerMixin):
    """Discriminant Correspondence Analysis (DISCA)

    Performance Discriminant Correspondence Analysis

    Parameters:
    ----------
    n_components:
    target :

    
    
    """
    def __init__(self,
                 n_components = None,
                 target = list[str],
                 features_labels=None,
                 mod_labels = None,
                 matrix_type = "completed",
                 priors = None,
                 parallelize = False):
        
        self.n_components = n_components
        self.target = target
        self.features_labels = features_labels
        self.mod_labels = mod_labels
        self.matrix_type = matrix_type
        self.priors = priors
        self.parallelize = parallelize
    
    def fit(self,X):


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

        
        self.target_ = self.target
        if self.target_ is None:
            raise ValueError("Error :'target' must be assigned.")

        self._computed_stats(X=X)
        

        return self
    
    def _global_stats(self,X,y):
        """Compute global statistiques of relations between two categorical variables

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Feature

        y : DataFrame of shape (n_samples,)
            Target
        """

        # Chi2 squared test - Tschuprow's T
        chi2_test = loglikelihood_test = pd.DataFrame(columns=["statistic","df","pvalue"],index=X.columns).astype("float")
        cramers_v = tschuprow_t = pearson= pd.DataFrame(columns=["value"],index=X.columns).astype("float")
        for cols in X.columns:
            # Crosstab
            tab = pd.crosstab(y[self.target_[0]],X[cols])

            # Chi2 - test
            chi2 = st.chi2_contingency(tab,correction=False)
            chi2_test.loc[cols,:] = np.array([chi2.statistic,chi2.dof,chi2.pvalue])

            # log-likelihood test
            loglikelihood = st.chi2_contingency(tab,lambda_="log-likelihood")
            loglikelihood_test.loc[cols,:] = np.array([loglikelihood.statistic,loglikelihood.dof,loglikelihood.pvalue])

            # Cramer's V
            cramers_v.loc[cols,:] = st.contingency.association(tab,method="cramer")

            # Tschuprow T statistic
            tschuprow_t.loc[cols,:] = st.contingency.association(tab,method="tschuprow")

            # Pearson
            pearson.loc[cols,:] = st.contingency.association(tab,method="pearson")
        
        quali_test = dict({"chi2" : chi2_test,
                           "log-likelihood-test":loglikelihood_test,
                           "cramer's V":cramers_v,
                           "tschuprow's T":tschuprow_t,
                           "pearson":pearson})
        
        return quali_test
    
    def _is_completed(self,X):
        """
        
        """

        # Continuous variables
        x = X.drop(columns=self.target_)
        # Qualitative variables - target
        y = X[self.target_]

        # Features columns
        self.features_labels_ = self.features_labels
        if self.features_labels_ is None:
            self.features_labels_ = x.columns

        # Update x
        x = x[self.features_labels_]
        # New data
        X = pd.concat([x,y],axis=1)

        # Dimension normale
        self.n_rows_, self.n_features_ = x.shape

        # Initial prior - proportion of each element
        if self.priors is None:
            p_k = y.value_counts(normalize=True)
        else:
            p_k = pd.Series(self.priors,index=self.classes_)

        ## Qualitatives tests
        self.statistics_test_ = self._global_stats(X=x,y=y)

        # Tableau des indicatrices
        dummies = pd.concat((pd.get_dummies(X[cols],prefix=cols,prefix_sep='_') for cols in (X.columns if self.features_labels_ is None else self.features_labels_)),axis=1)

        # Labels des modalités
        self.mod_labels_ = self.mod_labels
        if self.mod_labels_ is None:
            self.mod_labels_ = dummies.columns

        # 
        self.priors_ = p_k
        self.dummies_ = dummies
        # Données originales
        self.data_ = X

        # tableau de contingence
        return pd.concat([y,dummies],axis=1).groupby(self.target_).sum()

    def _is_dummies(self,X):
        """
        
        
        """

        # Suppression de la colonne target
        x = X.drop(columns=self.target_)

        # Labels des modalités
        self.mod_labels_ = self.mod_labels
        if self.mod_labels_ is None:
            self.mod_labels_ = x.columns

        # Qualitative variables - target
        y = X[self.target_]

        # Reconstitution de la matrice initiale
        data = from_dummies(data=x,sep="_")

        # Number of rows and features
        self.n_rows_, self.n_features_ = data.shape

        ## Qualitatives tests
        self.statistics_test_ = self._global_stats(X=data,y=y)

        # Labels des modalités
        self.features_labels_ = self.features_labels
        if self.features_labels_ is None:
            self.features_labels_ = data.columns

        # Initial prior - proportion of each element
        if self.priors is None:
            p_k = y.value_counts(normalize=True)
        else:
            p_k = pd.Series(self.priors,index=self.classes_)
        
        self.priors_ = p_k
        self.dummies_ = X.drop(columns=self.target_)

        # Données originales
        self.data_ = pd.concat([y,data],axis=1)

        # Matrice de contingence
        return X.groupby(self.target_).sum()

    def _computed_stats(self,X):

        """
        
        
        """

        if self.matrix_type == "completed":
            M = self._is_completed(X)
        elif self.matrix_type == "dummies":
            M = self._is_dummies(X)
        else:
            raise ValueError("Error : You must pass a valid 'matrix_type'.")
        

        
        # Les classes - groupes
        self.classes_ = M.index
        self.n_classes_ = len(self.classes_)
        #self.mod_labels_ = M.columns
        
        # Calcul des profils - Effectifs par classes (pour chaque descripteurs)
        n_l = M.sum(axis=0)
        # Profil marginal
        G = n_l/np.sum(n_l)
        
        mod_stats = pd.concat([n_l,G],axis=1)
        mod_stats.columns = ["n(l)","p(l)"]

        # Tableau des profils - Matric des profils
        profils = mapply(M,lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=self.n_workers_)
        
        # Distance entre un groupe et l'origine
        row_disto = mapply(profils,lambda x : np.sum((x-G.values)**2/G.values),axis=1,progressbar=False,n_workers=self.n_workers_).to_frame("disto(k)")

        # Distance entre les groupes - Matrice des distances par paires de classes
        row_dist = pd.DataFrame(squareform(pdist(profils,metric="seuclidean",V=G)**2),index=self.classes_,columns=self.classes_)

        # Inertie totale
        IT = np.sum([row_disto.loc[k]*self.priors_.loc[k,] for k in self.classes_])

        # Mise en oeuvre de l'AFC
        ca = CA(n_components=None,
                row_labels=M.index,
                col_labels=M.columns,
                row_sup_labels=None,
                col_sup_labels=None,
                parallelize=self.parallelize).fit(M)
            
        # Stockage des résultats de l'ACM
        col = get_ca_col(ca)

        # Coefficient des fonctions discriminantes canoniques
        coef = mapply(col["coord"],lambda x : x/(len(self.features_labels_)*np.sqrt(ca.eig_[0])),axis=1,progressbar=False,n_workers=self.n_workers_)
        
        # Coordonnées des individus à partir du tableau des indicatrices
        row_coord = self.dummies_.dot(coef)
        
        # Somme des carrés totales - totla sum of squared
        tss = mapply(row_coord,lambda x : x**2,axis=0,progressbar=False,n_workers=self.n_workers_).sum(axis=0)
        
        # Rapport de corrélation
        eta2 = ((self.n_rows_*ca.eig_[0])/tss).to_frame("correl. ratio").T

        # Coordonnées des classes
        gcoord = pd.DataFrame(ca.row_coord_,index=ca.row_labels_,columns=ca.dim_index_)

        # Qualite de la représentation des classes - COS2
        gcos2 = pd.DataFrame(ca.row_cos2_,index=ca.row_labels_,columns=ca.dim_index_)

        # Contribution des classes
        gcontrib = pd.DataFrame(ca.row_contrib_,index=ca.row_labels_,columns=ca.dim_index_)

        # Distance euclidienne entre les classes
        gdist = pd.DataFrame(ca.row_dist_,columns=ca.row_labels_,index=ca.row_labels_)

        # Informations sur les groupes
        ginfos = pd.DataFrame(ca.row_infos_,columns=ca.row_labels_,index=ca.row_labels_)

        ##### 
        # Coordonnées des modalités
        mod_coord = pd.DataFrame(ca.col_coord_,index=ca.col_labels_,columns=ca.dim_index_)

        # Cosinus carrés des modalités
        mod_cos2 = pd.DataFrame(ca.col_cos2_,index=ca.col_labels_,columns=ca.dim_index_)

        # Contributions des modalités
        mod_contrib = pd.DataFrame(ca.col_contrib_,index=ca.col_labels_,columns=ca.dim_index_)

        # Eigenvalues informations
        self.eig_ = ca.eig_
        self.n_components_ = ca.n_components_
        self.dim_index_ = ca.dim_index_

        # All informations
        self.row_dist_ = row_dist
        self.row_disto_ = row_disto
        self.row_coord_ = row_coord
        self.row_labels_ = X.index

        # Class informations
        self.gcoord_ = gcoord
        self.gdist_ = gdist
        self.gcos2_ = gcos2
        self.gcontrib_ = gcontrib
        self.ginfos_ = ginfos

        # Categories informations
        self.mod_stats_ = mod_stats
        self.mod_coord_ = mod_coord
        self.mod_cos2_ = mod_cos2
        self.mod_contrib_ = mod_contrib

        # Correspondance Analysis
        self.ca_model_ = ca

        # Inertie
        self.inertia_ = IT

        # Score function
        self.coef_ = coef
        
        # Correlation ratio
        self.correlation_ratio_ = eta2
        self.canonical_correlation_ratio_ = mapply(eta2,lambda x : np.sqrt(x),axis=0,progressbar=False,n_workers=self.n_workers_)

        self.model_ = "disca"

        
    def fit_transform(self,X):
        """
        
        
        
        """
        self.fit(X)
        return self.row_coord_
    

    def transform(self,X,y=None):
        """ Apply the dimensionality reduction on X. X is projected on
        the first axes previous extracted from a training set.
        Parameters
        ----------
        X : array of string, int or float, shape (n_rows_sup, n_vars)
            New data, where n_rows_sup is the number of supplementary
            row points and n_vars is the number of variables.
            X is a data table containing a category in each cell.
            Categories can be coded by strings or numeric values.
            X rows correspond to supplementary row points that are
            projected onto the axes.
        
        y : None
            y is ignored.
        Returns
        -------
        X_new : array of float, shape (n_rows_sup, n_components_)
            X_new : coordinates of the projections of the supplementary
            row points onto the axes.
        """
        if not isinstance(X,pd.DataFrame):
           raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        #self._compute_row_sup_stats(X)
        if self.matrix_type == "completed":
            n_rows = X.shape[0]
            n_cols = len(self.mod_labels_)
            Y = np.zeros((n_rows,n_cols))
            for i in np.arange(0,n_rows,1):
                values = [self.features_labels_[k] +"_"+str(X.iloc[i,k]) for k in np.arange(0,self.n_features_)]
                for j in np.arange(0,n_cols,1):
                    if self.mod_labels_[j] in values:
                        Y[i,j] = 1
            row_sup_dummies = pd.DataFrame(Y,columns=self.mod_labels_,index=X.index)
        elif self.matrix_type == "dummies":
            row_sup_dummies = X
        else:
            raise ValueError("Error : You must pass a valid 'matrix_type'.")
        
        return row_sup_dummies.dot(self.coef_)
    
    def decision_function(self,X):

        """Apply decision function to an array of samples.

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features)
            DataFrame of samples (test vectors).

        Returns
        -------
        C : DataFrame of shape (n_samples_,) or (n_samples_, n_classes)
            Decision function values related to each class, per sample.
        """

        # Coordonnées des individus
        coord = self.transform(X)

        # Distance euclidiennes aux centres de classes
        scores = pd.concat((mapply(self.gcoord_.sub(coord.loc[i,:].values,axis="columns"),lambda x : np.sum(x**2),axis=1,progressbar=False,n_workers=self.n_workers_).to_frame(i).rename_axis(None).T 
                            for i in coord.index),axis=0)

        return scores

    def predict_proba(self,X):
        """Estimate probability

        Parameters
        ----------
        X : DataFrame of shape (n_samples_,n_features_)
            Input data.
        
        Returns:
        --------
        C : DataFrame of shape (n_samples_,n_classes_)
            Estimated probabilities.
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        # Decision
        scores = self.decision_function(X)

        # Distance généralisée : scores - 2*log(p_k)
        DG = scores.sub((2*np.log(self.priors_.to_frame(name="p(k)").T.loc[:,scores.columns].values)),axis="columns")
    
        # Probabilité d'appartenance - transformation 
        C = mapply(mapply(DG,lambda x : np.exp(-0.5*x),axis=0,progressbar=False,n_workers=self.n_workers_),
                   lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=self.n_workers_)
        return C
    
    def predict(self,X):
        """Predict class labels for samples in X

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            The data matrix for which we want to get the predictions.
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples)
            Vectors containing the class labels for each sample
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        predict_proba = self.predict_proba(X)
        predict = np.unique(self.classes_)[np.argmax(predict_proba.values,axis=1)]
        predict = pd.DataFrame(predict,columns=["predict"],index=X.index)
        return predict
    
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