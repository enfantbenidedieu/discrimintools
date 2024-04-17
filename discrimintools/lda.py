# -*- coding: utf-8 -*-
import scipy.stats as st
import numpy as np
import pandas as pd
from functools import reduce, partial
from statsmodels.multivariate.manova import MANOVA
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

class LDA(BaseEstimator,TransformerMixin):
    """
    Linear Discriminant Analysis (LDA)
    ----------------------------------

    Description
    -----------

     Develops a discriminant criterion to classify each observation into groups

     Parameters:
     ----------

    distribution : {'multinomiale','homoscedastique'}
    priors : array-like of shape (n_classes,), default = None
        The class prior probabilities. By default, the class proportions are inferred from training data.
    
        
    Returns
    ------
    coef_ : DataFrame of shape (n_features,n_classes_)

    intercept_ : DataFrame of shape (1, n_classes)
    
    """
    def __init__(self,
                 features_labels=None,
                 target=list[str],
                 distribution = "homoscedastik",
                 row_labels = None,
                 priors = None,
                 parallelize = False):
        self.features_labels = features_labels
        self.target = target
        self.distribution = distribution
        self.row_labels = row_labels
        self.priors = priors
        self.parallelize = parallelize
    
    def fit(self,X,y=None):
        """Fit the Linear Discriminant Analysis model

        Parameters
        ----------
        X : DataFrame,
            Training Data
        
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

        
        self.data_ = X
        
        self.target_ = self.target
        if self.target_ is None:
            raise ValueError("Error :'target' must be assigned.")

        self._computed_stats(X=X)
        
        return self
    
    def _computed_stats(self,X):
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

        # Compute mean and standard deviation
        mean_std_var = x.agg(func = ["mean","std"])

        # categories
        self.classes_ = np.unique(y)

        # Number of rows and continuous variables
        self.n_samples_, self.n_features_ = x.shape

        # Number of groups
        self.n_classes_ = len(self.classes_)

        # Set row labels
        self.row_labels_ = self.row_labels
        if self.row_labels_ is None:
            self.row_labels = ["row."+str(i+1) for i in np.arange(0,self.n_samples_)]

        
        # Compute univariate ANOVA
        univariate_test = pd.DataFrame(np.zeros((self.n_features_,5)),index=self.features_labels_,
                                       columns=["Std. Dev.","R-squared","Rsq/(1-Rsq)","F-statistic","Prob (F-statistic)"])
        univariate_anova = dict()
        for lab in self.features_labels_:
            model = smf.ols(formula="{}~C({})".format(lab,"+".join(self.target_)), data=X).fit()
            univariate_test.loc[lab,:] = self.univariate_test_statistics(mean_std_var.loc["std",lab],model)
            univariate_anova[lab] = self.anova_table(sm.stats.anova_lm(model, typ=2))

        # Compute MULTIVARIATE ANOVA - MANOVA Test
        manova = MANOVA.from_formula(formula="{}~{}".format(paste(self.features_labels_,collapse="+"),"+".join(self.target_)), data=X).mv_test(skip_intercept_test=True)

        # Tukey Honestly significant difference - univariate
        tukey_test = dict()
        for name in self.features_labels_:
            comp = mc.MultiComparison(x[name],y[self.target_[0]])
            post_hoc_res = comp.tukeyhsd()
            tukey_test[name] = post_hoc_res.summary()

        # Bonferroni correction
        bonf_test = dict()
        for name in self.features_labels_:
            comp = mc.MultiComparison(x[name],y[self.target_[0]])
            tbl, a1, a2 = comp.allpairtest(st.ttest_ind, method= "bonf")
            bonf_test[name] = tbl
        
        # Sidak Correction
        sidak_test = dict()
        for name in self.features_labels_:
            comp = mc.MultiComparison(x[name],y[self.target_[0]])
            tbl, a1, a2 = comp.allpairtest(st.ttest_ind, method= "sidak")
            sidak_test[name] = tbl

        # Summary information
        summary_infos = pd.DataFrame({
            "Total Sample Size" : self.n_samples_,
            "Variables" : self.n_features_,
            "Classes" : self.n_classes_,
            "DF Total" : self.n_samples_ - 1,
            "DF Within Classes" : self.n_samples_ - self.n_classes_,
            "DF Between Classes" : self.n_classes_-1
        },index=["value"]).T

         # Rapport de correlation - Correlation ration
        eta2_res = dict()
        for name in self.features_labels_:
            eta2_res[name] = eta2(y,x[name])
        eta2_res = pd.DataFrame(eta2_res).T


        # Effectif par classe
        I_k = y.value_counts(normalize=False)

        # Initial prior - proportion of each element
        if self.priors is None:
            p_k = y.value_counts(normalize=True)
        else:
            p_k = pd.Series(self.priors,index=self.classes_)

        # Class level information
        class_level_information = pd.concat([I_k,p_k],axis=1,ignore_index=False)
        class_level_information.columns = ["n(k)","p(k)"]
        

        # Mean by group
        g_k = X.groupby(self.target_).mean()

        # Covariance totale
        V = x.cov(ddof=1)

        # Variance - Covariance par groupe - Matrices de covariance conditionnelles
        V_k = X.groupby(self.target_).cov(ddof=1)

        # Matrice de variance covariance intra - classe (corrivé en tenant compte du nombre de catégorie)
        W = list(map(lambda k : (I_k[k]-1)*V_k.loc[k],self.classes_))
        W = (1/(self.n_samples_-self.n_classes_))*reduce(lambda i,j : i + j, W)

        # Matrice de Variance Covariance inter - classe, obtenue par différence
        B = V - W
        
        # global performance
        self.global_performance_ = self._global_performance(V=V,W=W)

        # F - Exclusion - Statistical Evaluation
        self.statistical_evaluation_ = self._f_exclusion(V=V,W=W)

         # Squared Mahalanobis distances between class means
        self._mahalanobis_distances(X=W,y=g_k)

        if self.distribution == "homoscedastik":
            self._homoscedastik(W=W,gk=g_k,pk=p_k)

        # Store all information                                       # Mean in each group
        self.correlation_ratio_ = eta2_res
        self.summary_information_ = summary_infos
        self.class_level_information_ = class_level_information
        self.univariate_test_statistis_ = univariate_test
        self.anova_ = pd.concat(univariate_anova,axis=0) 
        self.manova_ = manova
        self.tukey_ = tukey_test
        self.bonferroni_correction_ = bonf_test
        self.sidak_ = sidak_test
        self.summary_information_ = summary_infos
        self.priors_ = p_k
        self.tcov_ = V
        self.bcov_ = B
        self.wcov_ = W
        self.gcov_ = V_k
        self.mean_ = mean_std_var.loc["mean",:]
        self.std_ = mean_std_var.loc["std",:]
        self.gmean_ = g_k

        # Distance aux centres de classes
        self.generalized_distance(X=x)


        self.model_ = "lda"

    def _homoscedastik(self,W,gk,pk):
        ##########################################################################################
        #           Fonctions de classement linaire
        ############################################################################################

        # Inverse de la matrice de variance - covariance intra - class
        invW = np.linalg.inv(W)

        # Calcul des coeffcients des variabes - features
        coef = gk.dot(invW).rename_axis(None).T
        coef.index = self.features_labels_

        # Constantes
        u = np.log(pk)
        b = gk.dot(invW)
        b.columns = self.features_labels_
        b = (1/2)*b.dot(gk.T)

        intercept = pd.DataFrame(dict({ k : u.loc[k,]-b.loc[k,k] for k in self.classes_}),index=["Intercept"])

        self.coef_ = coef
        self.intercept_ = intercept

    def _mahalanobis_distances(self,X,y):
        """
        Compute the Mahalanobis squared distance

        Parameters
        ----------
        X : pd.DataFrame.
            The
        
        y : pd.Series

        
        """

        # Matrice de covariance intra - classe utilisée par Mahalanobis
        W= X

        # Invesion
        invW = pd.DataFrame(np.linalg.inv(W),index=W.index,columns=W.columns)

        disto = pd.DataFrame(np.zeros((self.n_classes_,self.n_classes_)),index=self.classes_,columns=self.classes_)
        for i in np.arange(0,self.n_classes_-1):
            for j in np.arange(i+1,self.n_classes_):
                # Ecart entre les 2 vecteurs moyennes
                ecart = y.iloc[i,:] - y.iloc[j,:]
                # Distance de Mahalanobis
                disto.iloc[i,j] = np.dot(np.dot(ecart,invW),np.transpose(ecart))
                disto.iloc[j,i] = disto.iloc[i,j]
        
        self.squared_mdist_ = disto
    
    @staticmethod
    def anova_table(aov):
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        return aov
    
    @staticmethod
    def univariate_test_statistics(x,res):
        """
        Compute univariate Test Statistics

        Parameters
        ----------
        x : float. 
            Total Standard Deviation
        res : OLSResults.
            Results class for for an OLS model.
        
        Return
        -------
        univariate test statistics

        """

        return np.array([x,res.rsquared,res.rsquared/(1-res.rsquared),res.fvalue,res.f_pvalue])
    
    def generalized_distance(self,X):
        """Compute Generalized Distance
        
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        gen_dist = pd.DataFrame(columns=self.classes_,index=X.index).astype("float")
        for g in self.classes_:
            ecart =  X.sub(self.gmean_.loc[g].values,axis="columns")
            Y = np.dot(np.dot(ecart,np.linalg.inv(self.wcov_)),ecart.T)
            gen_dist.loc[:,g] = np.diag(Y) - 2*np.log(self.priors_.loc[g,])
        
        self.gen_dist_ = gen_dist
        
        return gen_dist

    def _global_performance(self,V,W):
        """Compute Global statistic - Wilks' Lambda - Bartlett statistic and Rao

        Parameters:
        ----------
        V : 
        
        Returns:
        --------

        
        """

        # Wilks' Lambda
        biased_V = ((self.n_samples_ - 1)/self.n_samples_)*V
        biased_W = ((self.n_samples_ - self.n_classes_)/self.n_samples_)*W
        
        # Lambda de Wilks
        lw = np.linalg.det(biased_W)/np.linalg.det(biased_V)

        ## Bartlett Test
        # Statistique B de Bartlett
        LB = -(self.n_samples_ - 1 - ((self.n_features_ + self.n_classes_)/2))*np.log(lw)
        # Degré de liberté
        ddl = self.n_features_*(self.n_classes_ - 1)
        
        ## RAO test
        # Valeur de A
        A = self.n_samples_ - self.n_classes_ - (1/2)*(self.n_features_ - self.n_classes_ + 2)
        
        # Valeur de B
        B = self.n_features_**2 + (self.n_classes_ - 1)**2 - 5
        if B > 0 :
            B = np.sqrt(((self.n_features_**2)*((self.n_classes_ - 1)**2)-4)/(B))
        else:
            B = 1

        # Valeur de C
        C = (1/2)*(self.n_features_*(self.n_classes_ - 1)-2)
        # statistic de test
        frao = ((1-(lw**(1/B)))/(lw**(1/B)))*((A*B-C)/(self.n_features_*(self.n_classes_ - 1)))

        # ddl numérateur
        ddlnum = self.n_features_*(self.n_classes_ - 1)
        # ddl dénominateur
        ddldenom = A*B-C
        
        # Resultat
        res = pd.DataFrame({"Stat" : ["Wilks' Lambda",f"Bartlett -- C({int(ddl)})",f"Rao -- F({int(ddlnum)},{int(ddldenom)})"],
                            "Value" : [lw,LB,frao],
                            "p-value": [np.nan,1 - st.chi2.cdf(LB,ddl),1 - st.f.cdf(frao,ddlnum,ddldenom)]})
        return res
    
    def _f_exclusion(self,V,W):

        """
        
        
        """

        # Wilks' Lambda
        biased_V = ((self.n_samples_ - 1)/self.n_samples_)*V
        biased_W = ((self.n_samples_ - self.n_classes_)/self.n_samples_)*W

        # Lambda de Wilks
        lw = np.linalg.det(biased_W)/np.linalg.det(biased_V)

        def fexclusion(j,W,V,n,K,lw):
            J = W.shape[1]
            # degrés de liberté - numérateur
            ddlsuppnum = K - 1
            # degrés de liberté dénominateur
            ddlsuppden = n - K - J + 1
            # Matrices intermédiaires numérateur
            tempnum = W.copy().values
            # Supprimer la référence de la variable à traiter
            tempnum = np.delete(tempnum, j, axis = 0)
            tempnum = np.delete(tempnum, j, axis = 1)
            # Même chose pour le numérateur
            tempden = V.values
            tempden = np.delete(tempden, j, axis = 0)
            tempden = np.delete(tempden, j, axis = 1)
            # Lambda de Wilk's sans la variable
            lwVar = np.linalg.det(tempnum)/np.linalg.det(tempden)
            # FValue
            fvalue = ddlsuppden/ddlsuppnum * (lwVar/lw-1)
            # Récupération des résultats
            return np.array([lwVar,lw/lwVar,fvalue,1 - st.f.cdf(fvalue, ddlsuppnum, ddlsuppden)])
        
        # Degré de liberté du numérateur
        ddl1 = self.n_classes_ - 1
        # Degré de liberté du dénominateur
        ddl2 = self.n_samples_ - self.n_classes_ - self.n_features_ +1 
        fextract = partial(fexclusion,W=biased_W,V=biased_V,n=self.n_samples_,K=self.n_classes_,lw=lw)
        res_contrib = pd.DataFrame(np.array(list(map(lambda j : fextract(j=j),np.arange(self.n_features_)))),
                                   columns=["Wilks L.","Partial L.",f"F{(ddl1,ddl2)}","p-value"],
                                   index= self.features_labels_)
            
        return res_contrib


    
    def decision_function(self,X):

        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. 

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features)
            DataFrame of samples (test vectors).

        Returns
        -------
        C : DataFrame of shape (n_samples_,) or (n_samples_, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is (n_samples_,), giving the
            log likelihood ratio of the positive class.
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.distribution == "multinomiale":
            scores = pd.DataFrame(columns=self.classes_,index=X.index).astype("float")
            for g in self.classes_:
                ecart =  X.sub(self.gmean_.loc[g].values,axis="columns")
                Y = np.dot(np.dot(ecart,np.linalg.inv(self.gcov_.loc[g])),ecart.T)
                scores.loc[:,g] = np.log(self.priors_.loc[g,]) - (1/2)*np.log(np.linalg.det(self.gcov_.loc[g,:]))-(1/2)*np.diag(Y)
        elif self.distribution == "homoscedastik":
            scores = X.dot(self.coef_).add(self.intercept_.values,axis="columns")
            scores.index = X.index
        return scores
    
    def transform(self,X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            Input data
        
        Returns:
        --------
        X_new : DataFrame of shape (n_samples_, n_components_)
            Transformed data.
        """

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        if self.distribution == "homoscedastik":
            predict = np.apply_along_axis(arr=np.dot(X,self.coef_),func1d=lambda x : x + self.intercept_,axis=1)
            return pd.DataFrame(predict,index=X.index,columns=self.dim_index_)

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
        # Probabilité d'appartenance - transformation 
        C = mapply(mapply(scores,lambda x : np.exp(x),axis=0,progressbar=False,n_workers=self.n_workers_),
                   lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=self.n_workers_)
        C.columns = self.classes_
        return C

    def fit_transform(self,X):
        """Fit to data, then transform it

        Fits transformer to `x` and returns a transformed version of X.

        Parameters:
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            Input samples
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_features_)
            Transformed data.
        
        
        """
        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        self.fit(X)

        return pd.DataFrame(self.row_coord_,index=X.index,columns=self.dim_index_)
    
    def score(self,X,y,sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features)
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

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)