# -*- coding: utf-8 -*-
import scipy.stats as st
import numpy as np
import pandas as pd
import polars as pl
from functools import reduce, partial
from statsmodels.multivariate.manova import MANOVA

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

from .eta2 import eta2

class LDA(BaseEstimator,TransformerMixin):
    """
    Linear Discriminant Analysis (LDA)
    ----------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Develops a discriminant criterion to classify each observation into groups

    Parameters:
    ----------

    target : The values of the classification variable define the groups for analysis.

    features : list of quantitative variables to be included in the analysis. The default is all numeric variables in dataset

    priors : The priors statement specifies the prior probabilities of group membership.
                - "equal" to set the prior probabilities equal,
                - "proportional" or "prop" to set the prior probabilities proportional to the sample sizes
                - a pandas series which specify the prior probability for each level of the classification variable.
        
    Returns
    ------
    call_ : a dictionary with some statistics

    coef_ : DataFrame of shape (n_features,n_classes_)

    intercept_ : DataFrame of shape (1, n_classes)

    summary_information_ :  summary information about the variables in the analysis. This information includes the number of observations, 
                            the number of quantitative variables in the analysis, and the number of classes in the classification variable. 
                            The frequency of each class is also displayed.

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates)

    statistics_ : statistics

    classes_ : classes informations

    cov_ : covariances

    model_ : string. The model fitted = 'lda'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    SAS Documentation, https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.3/statug/statug_discrim_overview.htm
    Ricco Rakotomalala, Pratique de l'analyse discriminante linéaire, Version 1.0, 2020
    """
    def __init__(self,
                 target = None,
                 features = None,
                 priors = None):
        self.target = target
        self.features = features
        self.priors = priors
    
    def fit(self,X,y=None):
        """
        Fit the Linear Discriminant Analysis model
        -------------------------------------------

        Parameters
        ----------
        X : pandas/polars DataFrame,
            Training Data
        
        Returns:
        --------
        self : object
            Fitted estimator
        """        
        # Mahalanobis distances
        def mahalanobis_distances(wcov,gmean,classes):
            """
            Compute the Mahalanobis squared distance
            ----------------------------------------

            Parameters
            ----------
            wcov : pandas dataframe, within covariance matrix
            
            gmean : pandas dataframe, conditional mean

            Return
            ------
            dist2 : square mahalanobis distance
            """
            # Invesion
            invW = pd.DataFrame(np.linalg.inv(wcov),index=wcov.index,columns=wcov.columns)
            n_classes = len(classes)

            dist2 = pd.DataFrame(np.zeros((n_classes,n_classes)),index=classes,columns=classes)
            for i in np.arange(n_classes-1):
                for j in np.arange(i+1,n_classes):
                    # Ecart entre les 2 vecteurs moyennes
                    ecart = gmean.iloc[i,:] - gmean.iloc[j,:]
                    # Distance de Mahalanobis
                    dist2.iloc[i,j] = np.dot(np.dot(ecart,invW),np.transpose(ecart))
                    dist2.iloc[j,i] = dist2.iloc[i,j]
            
            return dist2
        
        # Anova analysis
        def anova_table(aov):
            aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
            aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
            aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
            cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
            aov = aov[cols]
            return aov
        
        # Univariate test statistics
        def univariate_test_statistics(stdev,model):
            """
            Compute univariate Test Statistics
            ----------------------------------

            Parameters
            ----------
            stdev : float. 
                Total Standard Deviation
            model : OLSResults.
                Results class for for an OLS model.
            
            Return
            -------
            univariate test statistics
            """
            return np.array([stdev,model.rsquared,model.rsquared/(1-model.rsquared),model.fvalue,model.f_pvalue])
        
        def generalized_distance(X,wcov,gmean,classes,priors):
            """
            Compute Generalized Distance
            ----------------------------

            Parameters
            ----------
            X : pandas dataframe of shape (n_rows, n_features)

            wcov : pandas dataframe, within covariance matrix

            gmean : pandas dataframe, conditional mean

            classes : class groups

            priors : Class priors (sum to 1)

            Return
            ------
            dist2 : square generalized distance
            """
            dist2 = pd.DataFrame(columns=classes,index=X.index).astype("float")
            for g in classes:
                ecart =  X.sub(gmean.loc[g].values,axis="columns")
                Y = np.dot(np.dot(ecart,np.linalg.inv(wcov)),ecart.T)
                dist2.loc[:,g] = np.diag(Y) - 2*np.log(priors.loc[g,])
            
            return dist2
        
        # Multivariate ANOVA
        def global_performance(V,W,n_samples):
            """
            Compute Global statistic - Wilks' Lambda - Bartlett statistic and Rao
            ---------------------------------------------------------------------

            Parameters
            ----------
            V : pandas dataframe, total covariance matrix

            W : pandas dataframe, within covariance matrix

            n_classes : number of classes
            
            Returns:
            --------
            manova : Multivariate Anova
            """

            # Set number of features
            n_features = V.shape[0]

            # Wilks' Lambda
            biased_V = ((n_samples - 1)/n_samples)*V
            biased_W = ((n_samples - n_classes)/n_samples)*W
            
            # Lambda de Wilks
            lw = np.linalg.det(biased_W)/np.linalg.det(biased_V)

            ## Bartlett Test
            # Statistique B de Bartlett
            LB = -(n_samples - 1 - ((n_features + n_classes)/2))*np.log(lw)
            # Degré de liberté
            ddl = n_features*(n_classes - 1)
            
            ## RAO test
            # Valeur de A
            A = n_samples - n_classes - (1/2)*(n_features - n_classes + 2)
        
            # Valeur de B
            B = n_features**2 + (n_classes - 1)**2 - 5
            if B > 0 :
                B = np.sqrt(((n_features**2)*((n_classes - 1)**2)-4)/(B))
            else:
                B = 1

            # Valeur de C
            C = (1/2)*(n_features*(n_classes - 1)-2)
            # statistic de test
            frao = ((1-(lw**(1/B)))/(lw**(1/B)))*((A*B-C)/(n_features*(n_classes - 1)))

            # ddl numérateur
            ddlnum = n_features*(n_classes - 1)
            # ddl dénominateur
            ddldenom = A*B-C
            
            # Resultat
            manova = pd.DataFrame({"Stat" : ["Wilks' Lambda",f"Bartlett -- C({int(ddl)})",f"Rao -- F({int(ddlnum)},{int(ddldenom)})"],
                                "Value" : [lw,LB,frao],
                                "p-value": [np.nan,1 - st.chi2.cdf(LB,ddl),1 - st.f.cdf(frao,ddlnum,ddldenom)]})
            return manova
        
        ##### Pertinance d'un group de variables
        def f_exclusion(n_samples,n_classes,n_features,V,W):
            # Wilks' Lambda
            biased_V = ((n_samples - 1)/n_samples)*V
            biased_W = ((n_samples - n_classes)/n_samples)*W

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
            ddl1 = n_classes - 1
            # Degré de liberté du dénominateur
            ddl2 = n_samples - n_classes - n_features +1 
            fextract = partial(fexclusion,W=biased_W,V=biased_V,n=n_samples,K=n_classes,lw=lw)
            res_contrib = pd.DataFrame(np.array(list(map(lambda j : fextract(j=j),np.arange(n_features)))),
                                    columns=["Wilks L.","Partial L.",f"F{(ddl1,ddl2)}","p-value"],
                                    index= V.index)
                
            return res_contrib
            
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

        # Compute statistiques for quantitatives variables
        stats = x.describe().T
        
        # Compute univariate ANOVA
        univ_test = pd.DataFrame(np.zeros((n_features,5)),index=x.columns,columns=["Std. Dev.","R-squared","Rsq/(1-Rsq)","F-statistic","Prob (F-statistic)"])
        anova = {}
        for lab in x.columns:
            model = smf.ols(formula="{}~C({})".format(lab,"+".join(self.target)), data=X).fit()
            univ_test.loc[lab,:] = univariate_test_statistics(stdev= stats.loc[lab,"std"],model=model)
            anova[lab] = anova_table(sm.stats.anova_lm(model, typ=2))
        anova = pd.concat(anova)
        
        # Rapport de correlation - Correlation ration
        eta2_res = {}
        for col in x.columns:
            eta2_res[col] = eta2(y,x[col])
        eta2_res = pd.DataFrame(eta2_res).T

        # Compute MULTIVARIATE ANOVA - MANOVA Test
        manova = MANOVA.from_formula(formula="{}~{}".format("+".join(x.columns),"+".join(self.target)), data=X).mv_test(skip_intercept_test=True)

        statistics = {"anova" : anova,"manova" : manova,"Eta2" : eta2_res,"univariate" : univ_test}

        # Summary information
        summary_infos = pd.DataFrame({
            "infos" : ["Total Sample Size","Variables","Classes"],
            "Value" : [n_samples,n_features,n_classes],
            "DF" : ["DF Total", "DF Within Classes", "DF Between Classes"],
            "DF value" : [n_samples-1,n_samples - n_classes, n_classes-1]
        })
        self.summary_information_ = summary_infos

        # Number of element 
        n_k, p_k = y.value_counts(normalize=False), y.value_counts(normalize=True)

        # Initial prior - proportion of each element
        if self.priors is None:
            raise ValueError("'priors' must be assigned")
        
        if isinstance(self.priors,pd.Series):
            priors = pd.Series([x/self.priors.sum() for x in self.priors.values],index=self.priors.index)
        elif isinstance(self.priors,str):
            if self.priors in ["proportional","prop"]:
                priors = p_k
            elif self.priors == "equal":
                priors = pd.Series([1/n_classes]*n_classes,index=classes)
            
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

        # Conditional mean by class
        g_k = X.groupby(self.target).mean()

        # Covariance totale biaisée
        V = x.cov(ddof=1)

        # Variance - Covariance par groupe - Matrices de covariance conditionnelles
        V_k = X.groupby(self.target).cov(ddof=1)

        # Matrice de variance covariance intra - classe (corrivé en tenant compte du nombre de catégorie)
        W = list(map(lambda k : (n_k[k]-1)*V_k.loc[k],classes))
        W = (1/(n_samples-n_classes))*reduce(lambda i,j : i + j, W)

        # Matrice de Variance Covariance inter - classe, obtenue par différence
        B = V - W

        ################################################################"
        # Rank of pooled covariance matrix
        rankW = np.linalg.matrix_rank(W)
        logDetW = np.log(np.linalg.det(W))
        statistics["pooled_information"] = pd.DataFrame([rankW,logDetW],columns=["value"],
                                                        index=["Rang de la mat. de cov. intra-classes",
                                                               "Log. naturel du det. de la mat. de cov. intra-classes"])
        
        ###########################################################################################################
        # Coefficients des fonctions de classement
        ###########################################################################################################

        # Inverse de la matrice de variance - covariance intra - class
        invW = np.linalg.inv(W)

        # Calcul des coeffcients des variabes - features
        coef = g_k.dot(invW).rename_axis(None).T
        coef.index = x.columns

        # Constantes
        u = np.log(priors)
        b = g_k.dot(invW)
        b.columns = x.columns
        b = (1/2)*b.dot(g_k.T)

        intercept = pd.DataFrame({k : u.loc[k,]-b.loc[k,k] for k in classes},index=["Intercept"])

        self.coef_ = coef
        self.intercept_ = intercept

        ##########################################################################################################
        # Appartenance à la classe - Scores
        scores = x.dot(coef).add(intercept.values,axis="columns")

        ### Generalized distance - Distance aux centres de classes
        gen_dist2 = generalized_distance(X=x,wcov=W,gmean=g_k,classes=classes,priors=priors)

        self.ind_ = {"scores" : scores, "generalized_dit2" : gen_dist2}
       
        ############################################################################################################
        # Performance du modèle
        #############################################################################################################   
    
        # global performance
        global_perf = global_performance(V=V,W=W,n_samples=n_samples)
        statistics["performance"] = global_perf

        # F - Exclusion - Statistical Evaluation
        statistical_evaluation = f_exclusion(n_samples=n_samples,n_classes=n_classes,n_features=n_features,V=V,W=W)
        statistics["statistical_evaluation"] = statistical_evaluation
        self.statistics_ = statistics

        # Squared Mahalanobis distances between class means
        squared_mdist2 = mahalanobis_distances(wcov=W,gmean=g_k,classes=classes)

        self.classes_ = {"classes" : classes,"mean" : g_k,"cov" :V_k, "mahalanobis" : squared_mdist2}
        self.cov_ = {"total" : V , "within" : W, "between" : B}    
        
        self.model_ = "lda"
        
        return self
    
    def transform(self,X):
        """
        Project data to maximize class separation
        -----------------------------------------

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            Input data
        
        Returns:
        --------
        X_new : DataFrame of shape (n_samples_, n_classes_)
            Transformed data.
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
        return X.dot(self.coef_).add(self.intercept_.values,axis="columns")
    
    def fit_transform(self,X):
        """
        Fit to data, then transform it
        ------------------------------

        Fits transformer to `x` and returns a transformed version of X.

        Parameters:
        ----------
        X : DataFrame of shape (n_samples_, n_features_+1)
            Input samples
        
        Returns
        -------
        X_new : DataFrame of shape (n_rows, n_classes_)
            Transformed data.
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        self.fit(X)
        return self.ind_["scores"]
    
    def decision_function(self,X):
        """
        Apply decision function to an array of samples
        ----------------------------------------------

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
        return  X.dot(self.coef_).add(self.intercept_.values,axis="columns")
    
    def predict_proba(self,X):
        """
        Estimate probability
        --------------------

        Parameters
        ----------
        X : DataFrame of shape (n_samples_,n_features_)
            Input data.
        
        Returns:
        --------
        C : DataFrame of shape (n_samples_,n_classes_)
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
        # Decision
        scores = self.decision_function(X)
        return scores.apply(lambda x : np.exp(x),axis=0).apply(lambda x : x/np.sum(x),axis=1)
    
    def predict(self,X):
        """
        Predict class labels for samples in X
        --------------------------------------

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
        predict = np.unique(self.classes_["classes"])[np.argmax(predict_proba.values,axis=1)]
        return pd.Series(predict,index=X.index,name="prediction")

    def score(self,X,y,sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels
        ----------------------------------------------------------

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
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

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