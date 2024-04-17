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

#######################################################################################################
#           Discriminant Qualitatives (DISQUAL)
#######################################################################################################


class DISQUAL(BaseEstimator,TransformerMixin):

    """Discriminant Qualitatives

    Performs discriminant analysis for categorical variables using multiple correspondence analysis (MCA)

    Parameters:
    ----------
    n_components : 
    target :
    features_labels :
    row_labels :
    priors : 

    Returns:
    -------
    eig_ :
    mod_ :
    row_ :
    lda_model_ :
    mca_model_ :
    statistical_evaluation_ : 
    projection_function_ : 
    lda_coef_ :
    lda_intercept_ :
    coef_ :
    intercept_ :

    
    Note:
    -----
    - https://lemakistatheux.wordpress.com/category/outils-danalyse-supervisee/la-methode-disqual/
    - Probabilité, analyse des données et Statistique de Gilbert Saporta.
    - Data Mining et statistique décisionnelle de Stéphane Tufféry.

    #
    prodécure SAS: http://od-datamining.com/download/#macro
    Package et fonction R :
    http://finzi.psych.upenn.edu/library/DiscriMiner/html/disqual.html
    https://github.com/gastonstat/DiscriMiner


    
    
    """


    def __init__(self,
                 n_components = None,
                 target = list[str],
                 features_labels=None,
                 row_labels = None,
                 priors=None,
                 parallelize=False):
        
        self.n_components = n_components
        self.target = target
        self.features_labels = features_labels
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

        self._compute_stats(X=X)
        
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
            tab = pd.crosstab(X[cols],y[self.target_[0]])

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
    
    def _compute_stats(self,X):
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

        # categories
        self.classes_ = np.unique(y)

        # Number of observations
        self.n_samples_, self.n_features_ = x.shape

        # Number of groups
        self.n_classes_ = len(self.classes_)

        ## Qualitatives tests
        self.statistics_test_ = self._global_stats(X=x,y=y)

        # Set row labels
        self.row_labels_ = self.row_labels
        if self.row_labels_ is None:
            self.row_labels = ["row."+str(i+1) for i in np.arange(0,self.n_samples_)]

        # Analyse des correspondances multiples (MCA)
        mca = MCA(n_components=self.n_components,row_labels=self.row_labels_,var_labels=self.features_labels_,
                  mod_labels=None,matrix_type="completed",benzecri=False,greenacre=False,
                  row_sup_labels=None,quali_sup_labels=None,quanti_sup_labels=None,parallelize=self.parallelize).fit(x)
        
        # Stockage des résultats de l'ACM
        mod = get_mca_mod(mca)
        row = get_mca_ind(mca)

        # Fonction de projection
        fproj = mapply(mod["coord"],lambda x : x/(self.n_features_*np.sqrt(mca.eig_[0])),axis=1,progressbar=False)

        # Données pour l'Analyse Discriminante Linéaire
        row_coord = row["coord"]
        row_coord.columns = list(["Z"+str(x+1) for x in np.arange(0,mca.n_components_)])
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

        # Coefficient du DISCQUAL
        coef = pd.DataFrame(np.dot(fproj,lda_coef),index=fproj.index,columns=lda_coef.columns)
        
        # Sortie de l'ACM
        self.projection_function_ = fproj
        self.n_components_ = mca.n_components_

        # Sortie de l'ADL
        self.lda_coef_ = lda_coef
        self.lda_intercept_ = lda_intercept
        self.lda_features_labels_ = list(["Z"+str(x+1) for x in np.arange(0,mca.n_components_)])
        
        # Informations du DISCQUAL
        self.statistical_evaluation_ = lda.statistical_evaluation_
        self.coef_ = coef
        self.intercept_ = lda_intercept

        # Stockage des deux modèles
        self.mca_model_ = mca
        self.lda_model_ = lda

        self.model_ = "disqual"
    
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

        return self.mca_model_.row_coord_
    
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

        coord = self.mca_model_.transform(X)
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