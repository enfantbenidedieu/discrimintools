# -*- coding: utf-8 -*-
import scipy.stats as st
import numpy as np
import pandas as pd
import polars as pl
from functools import reduce
from scipy.spatial.distance import pdist,squareform
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, TransformerMixin
from mapply.mapply import mapply
from sklearn.metrics import accuracy_score

from .eta2 import eta2

##########################################################################################
#                       CANONICAL DISCRIMINANT ANALYSIS (CANDISC)
##########################################################################################

class CANDISC(BaseEstimator,TransformerMixin):
    """
    Canonical Discriminant Analysis (CANDISC)
    -----------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs a Canonical Discriminant Analysis, computes squared 
    Mahalanobis distances between class means, and performs both 
    univariate and multivariate one-way analyses of variance

    Parameters
    ----------
    n_components : number of dimensions kept in the results

    target : string, target variable

    priors : Class priors (sum to 1)

    parallelize : boolean, default = False
        If model should be parallelize
            - If True : parallelize using mapply
            - If False : parallelize using apply

    Return
    ------
    summary_information_ :  summary information about the variables in the analysis. This information includes the number of observations, 
                            the number of quantitative variables in the analysis, and the number of classes in the classification variable. 
                            The frequency of each class is also displayed.

    eig_  : a pandas dataframe containing all the eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    ind_ : a dictionary of pandas dataframe containing all the results for the active individuals (coordinates)

    statistics_ : statistics

    classes_ : classes informations

    cov_ : covariances

    corr_ : correlation

    coef_ : pandas dataframe, Weight vector(s).

    intercept_ : pandas dataframe, Intercept term.

    score_coef_ :

    score_intercept_ : 

    svd_ : eigenv value decomposition

    call_ : a dictionary with some statistics

    model_ : string. The model fitted = 'candisc'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    SAS Documentation, https://documentation.sas.com/doc/en/statug/15.2/statug_candisc_toc.htm
    https://www.rdocumentation.org/packages/candisc/versions/0.8-6/topics/candisc
    https://www.rdocumentation.org/packages/candisc/versions/0.8-6
    Ricco Rakotomalala, Pratique de l'analyse discriminante linéaire, Version 1.0, 2020
    """
    def __init__(self,
                 n_components=None,
                 target=None,
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
        Fit the Canonical Discriminant Analysis model
        ---------------------------------------------

        Parameters
        ----------
        X : pandas/polars DataFrame,
            Training Data
        
        Returns:
        --------
        self : object
            Fitted estimator
        """

        # Between pearson correlation
        def betweencorrcoef(g_k,z_k,name,lda,weights):
            def m(x, w):
                return np.average(x,weights=w)
            def cov(x, y, w):
                return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)
            def corr(x, y, w):
                return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))
            return corr(g_k[name], z_k[lda],weights)
        
        # Manlanobis distances
        def mahalanobis_distances(wcov,gmean,n_obs,classe):
            """
            Compute the Mahalanobis squared distance
            ----------------------------------------

            Parameters
            ----------
            wcov :  pandas dataframe, within covariance matrix 
            
            gmean :  pandas dataframe, conditional mean

            Return
            ------
            dist2 : square mahalanobie distance
            """
            # Number of class
            n_classes = len(classe)

            # Matrice de covariance intra - classe utilisée par Mahalanobis
            W = (n_obs/(n_obs - n_classes))*wcov

            # Invesion
            invW = pd.DataFrame(np.linalg.inv(W),index=W.index,columns=W.columns)

            dist2 = pd.DataFrame(np.zeros((n_classes,n_classes)),index=classe,columns=classe)
            for i in np.arange(0,n_classes-1):
                for j in np.arange(i+1,n_classes):
                    # Ecart entre les 2 vecteurs moyennes
                    ecart = gmean.iloc[i,:] - gmean.iloc[j,:]
                    # Distance de Mahalanobis
                    dist2.iloc[i,j] = np.dot(np.dot(ecart,invW),np.transpose(ecart))
                    dist2.iloc[j,i] = dist2.iloc[i,j]
    
            return dist2
        
        # Analysis of Variance table
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
            stdev : float. Total Standard Deviation

            model : OLSResults.Results class for for an OLS model.
            
            Return
            -------
            univariate test statistics
            """
            return np.array([stdev,model.rsquared,model.rsquared/(1-model.rsquared),model.fvalue,model.f_pvalue])
        
        # Global performance 
        def global_performance(n_features,n_obs,n_classes,lw):
            """
            Compute Global statistic - Wilks' Lambda - Bartlett statistic and Rao
            ---------------------------------------------------------------------

            Parameters:
            ----------
            lw : float
                Wilks lambda's value
            
            Returns:
            --------
            """
            #########################################################
            ## Bartlett Test
            # Statistique B de Bartlett & Degré de liberté
            B, ddl = -(n_obs - 1 - ((n_features + n_classes)/2))*np.log(lw), n_features*(n_classes - 1)
            
            ##############################################################
            ## RAO test

            # Valeur intermédiaire pour calcul du ddl dénominateur
            temp = n_features**2 + (n_classes - 1)**2 - 5
            temp = np.where(temp>0,np.sqrt(((n_features**2)*((n_classes - 1)**2)-4)/temp),1)
            # ddl dénominateur
            ddldenom = (2*n_obs - n_features - n_classes - 2)/2*temp - (ddl - 2)/2
            # statistic de test
            frao = ((1-(lw**(1/temp)))/(lw**(1/temp)))*(ddldenom/ddl)
            # Resultat
            res = pd.DataFrame({"Stat" : ["Wilks' Lambda",f"Bartlett -- C({int(ddl)})",f"Rao -- F({int(ddl)},{int(ddldenom)})"],
                                "Value" : [lw,B,frao],
                                "p-value": [np.nan,1 - st.chi2.cdf(B,ddl),1 - st.f.cdf(frao,ddl,ddldenom)]})
            return res
        
        def likelihood_test(n_samples,n_features,n_classes,eigen):
            # Statistique de test
            if isinstance(eigen,float) or isinstance(eigen,int):
                q = 1
            else:
                q = len(eigen)
            LQ = np.prod([(1-i) for i in eigen])
            # r
            r = n_samples - 1 - (n_features+n_classes)/2
            # t
            if ((n_features - n_classes + q + 1)**2 + q**2 - 5) > 0 : 
                t = np.sqrt((q**2*(n_features-n_classes+q+1)**2-4)/((n_features - n_classes + q + 1)**2 + q**2 - 5)) 
            else: 
                t = 1
            # u
            u = (q*(n_features-n_classes+q+1)-2)/4
            # F de RAO
            FRAO = ((1 - LQ**(1/t))/(LQ**(1/t))) * ((r*t - 2*u)/(q*(n_features - n_classes + q + 1)))
            # ddl1 
            ddl1 = q*(n_features - n_classes + q + 1)
            # ddl2
            ddl2 = r*t - 2*u
            res_rao = pd.DataFrame({"statistic":FRAO,"DDL num." : ddl1, "DDL den." : ddl2,"Pr>F": 1 - st.f.cdf(FRAO,ddl1,ddl2)},index=["test"])
            return res_rao

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

        ###################################### Set number of components ##########################################
        if self.n_components is None:
            n_components = min(n_features-1,n_classes - 1)
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,n_features-1,n_classes - 1)

        # Compute statistiques for quantitatives variables
        stats = x.describe().T

        # Compute univariate ANOVA
        univ_test = pd.DataFrame(np.zeros((n_features,5)),index=x.columns,columns=["Std. Dev.","R-squared","Rsq/(1-Rsq)","F-statistic","Prob (F-statistic)"])
        anova = {}
        for lab in x.columns:
            model = smf.ols(formula="{}~C({})".format(lab,"+".join(self.target)), data=X).fit()
            univ_test.loc[lab,:] = univariate_test_statistics(stdev=stats.loc[lab,"std"],model=model)
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

        # Number of eflemnt in each group
        n_k, p_k = y.value_counts(normalize=False),  y.value_counts(normalize=True)

         # Initial prior - proportion of each element
        if self.priors is None:
            raise ValueError("'priors' must be assigned")

        # Prior probability
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
                      "n_components" : n_components,
                      "priors" : p_k}
        
        #############################
        # Class level information
        class_level_information = pd.concat([n_k,p_k,priors],axis=1)
        class_level_information.columns = ["Frequency","Proportion","Prior Probability"]
        statistics["information"] = class_level_information
        
        # Mean by group
        g_k = X.groupby(self.target).mean()

        # Covariance totale
        V = x.cov(ddof=0)

        # Variance - Covariance par groupe
        V_k = X.groupby(self.target).cov(ddof=1)

        # Matrice de variance covariance intra - classe
        W = list(map(lambda k : (n_k[k]-1)*V_k.loc[k],classes))
        W = (1/n_samples)*reduce(lambda i,j : i + j, W)

        # Matrice de Variance Covariance inter - classe, obtenue par différence
        B = V - W

        # Squared Mahalanobis distances between class means
        squared_mdist2 = mahalanobis_distances(wcov=W,gmean=g_k,n_obs=n_samples,classe=classes)

        # First Matrix - French approach
        M1 = B.dot(np.linalg.inv(V)).T
        eigen1, _ = np.linalg.eig(M1)
        eigen_values1 = np.real(eigen1[:n_components])

        # Second Matrix - Anglosaxonne approach
        M2 = B.dot(np.linalg.inv(W)).T
        eigen2, _ = np.linalg.eig(M2)
        eigen_values2 = np.real(eigen2[:n_components])

        # Eigenvalue informations
        eigen_values = eigen_values2
        difference = np.insert(-np.diff(eigen_values),len(eigen_values)-1,np.nan)
        proportion = 100*eigen_values/np.sum(eigen_values)
        cumulative = np.cumsum(proportion)

        # Correction approach
        C = pd.concat(list(map(lambda k : np.sqrt(priors.loc[k].values)*g_k.loc[k,:].sub(stats.loc[:,"mean"],axis="index").to_frame(k),classes)),axis=1)
        
        # Diagonalisation de la matrice M
        M3 = np.dot(np.dot(C.T,np.linalg.inv(V)),C)
        eigen3, vector3 = np.linalg.eig(M3)
        # Gestion des nombres complexes
        eigen3 = np.real(eigen3)
        vector3 = np.real(vector3)

        # Reverse sort eigenvalues - Eigenvalues aren't in best order
        new_eigen = np.array(sorted(eigen3,reverse=True))
        idx = [list(eigen3).index(x) for x in new_eigen]

        # New eigen vectors
        eigen = new_eigen[:n_components]
        vector = vector3[:,idx][:,:n_components]
        
        #####################################################################################################
        #########################"" vecteur beta
        beta_l = np.dot(np.dot(np.linalg.inv(V),C),vector)
        beta_l = pd.DataFrame(beta_l,index=x.columns,columns= ["LD"+str(x+1) for x in np.arange(beta_l.shape[1])])

        ############## Coefficients 
        u_l = mapply(beta_l,lambda x : x*np.sqrt((n_samples-n_classes)/(n_samples*eigen*(1-eigen))),axis=1,progressbar=False,n_workers=n_workers)
        
        ##################### Intercept
        u_l0 = - u_l.T.dot(stats.loc[:,"mean"])
        u_l0.name = "intercept"

        ##############################################################################################################
        # Coordonnées des individus
        ##############################################################################################################
        row_coord = x.dot(u_l).add(u_l0,axis="columns")

        # Class Means on Canonical Variables
        gcoord = pd.concat((row_coord,y),axis=1).groupby(self.target).mean()

        # Coordonnées des centres de classes
        z_k = mapply(g_k.dot(u_l),lambda x : x + u_l0,axis=1,progressbar=False,n_workers=n_workers)

        # Distance entre barycentre
        disto = pd.DataFrame(squareform(pdist(z_k,metric="sqeuclidean")),columns=classes,index=classes)

        # Lambda de Wilks
        lw = np.linalg.det(W)/np.linalg.det(V)
        
        # global performance
        global_perf = global_performance(n_features=n_features,n_obs=n_samples,n_classes=n_classes,lw=lw)
        statistics["performance"] = global_perf

        # Test sur un ensemble de facteurs - likelohood ratio test
        lrt_test = pd.DataFrame(np.zeros((n_components,4)),columns=["statistic","DDL num.","DDL den.","Pr>F"]).astype("float")
        for i in np.arange(n_components):
            lrt_test.iloc[-i,:] = likelihood_test(n_samples=n_samples,n_features=n_features,n_classes=n_classes,eigen=eigen_values1[-(i+1):])
        lrt_test = lrt_test.sort_index(ascending=False).reset_index(drop=True)
        statistics["likelihood_test"] = lrt_test
        self.statistics_ = statistics

        ###############################################################################################################
        #  Correlation 
        ###############################################################################################################
        ################# Total correlation - Total Canonical Structure
        tcorr = np.corrcoef(x=x,y=row_coord,rowvar=False)[:x.shape[1],x.shape[1]:]
        tcorr = pd.DataFrame(tcorr,index=x.columns,columns= ["LD"+str(x+1) for x in np.arange(tcorr.shape[1])])

        ################# Within correlation - Polled Within Canonical Structure
        z1 = row_coord - gcoord.loc[y[self.target[0]],:].values
        g_k_long = g_k.loc[y[self.target[0]],:]
        z2 = pd.DataFrame(x.values - g_k_long.values,index=g_k_long.index,columns=x.columns)
        wcorr = np.transpose(np.corrcoef(x=z1,y=z2,rowvar=False)[:n_components,n_components:])
        wcorr = pd.DataFrame(wcorr,columns= ["LD"+str(x+1) for x in np.arange(wcorr.shape[1])],index=x.columns)

        ################# Between correlation - Between Canonical Structure
        bcorr = pd.DataFrame(np.zeros((n_features,n_components)),index=x.columns,columns=["LD"+str(x+1) for x in np.arange(n_components)])
        for name in x.columns:
            for name2 in bcorr.columns:
                bcorr.loc[name,name2]= betweencorrcoef(g_k,z_k,name,name2,priors.values)

        ###################################################################################################################
        # Fonction de classement
        #######################################################################################################################
        # Coefficients de la fonction de décision (w_kj, k=1,...,K; j=1,...,J)
        u_l_df = pd.DataFrame(u_l,index=x.columns,columns=["LD"+str(x+1) for x in np.arange(n_components)])
        S_omega_k = pd.DataFrame(map(lambda k :
                            pd.DataFrame(map(lambda l : u_l_df.loc[:,l]*z_k.loc[k,l],row_coord.columns)).sum(axis=0),classes), 
                            index = classes).T
        
        # Constante de la fonction de décision
        S_omega_k0 = pd.DataFrame(
                    map(lambda k : (np.log(p_k.loc[k,])+sum(u_l0.T*z_k.loc[k,:])-
                0.5*sum(z_k.loc[k,:]**2)),classes),index = classes,
                columns = ["intercept"]).T

        ###################################
        # Store all informations
        eig = np.c_[eigen_values[:n_components],difference[:n_components],proportion[:n_components],cumulative[:n_components]]
        self.eig_ = pd.DataFrame(eig,columns=["Eigenvalue","Difference","Proportion","Cumulative"],index = ["LD"+str(x+1) for x in range(eig.shape[0])])
         
        self.ind_ = {"coord" : row_coord}
        self.classes_ = {"classes" : classes,"coord" : gcoord,"mean" : g_k,"cov" :V_k,"center" : z_k ,"dist" : disto,"mahalanobis" : squared_mdist2}
        self.cov_ = {"total" : V , "within" : W, "between" : B}          
        self.corr_ = {"total" : tcorr, "within" : wcorr, "between" : bcorr}

        # Coefficients
        self.coef_ = u_l
        # Intercept
        self.intercept_ = u_l0

        # Fonction de classement - coefficient
        self.score_coef_ = S_omega_k
        self.score_intercept_ = S_omega_k0

        self.svd_ = {"value" : eigen_values[:n_components],"vector" : vector[:,:n_components]}

        # Model name
        self.model_ = "candisc"
        
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
        X_new : DataFrame of shape (n_samples_, n_components_)
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
        
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1
        
        ##### Chack if target in X columns
        if self.call_["target"] in X.columns.tolist():
            X = X.drop(columns=[self.call_["target"]])

        ####### Select features
        X = X[self.call_["features"]]
        
        coord = mapply(X.dot(self.coef_),lambda x : x + self.intercept_.values,axis=1,progressbar=False,n_workers=n_workers)
        return coord
    
    def fit_transform(self,X):
        """
        Fit to data, then transform it
        ------------------------------

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
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
        self.fit(X)
        return self.ind_["coord"]
    
    def decision_function(self,X):
        """
        Apply decision function to a pandas dataframe of samples
        --------------------------------------------------------

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
        return X.dot(self.score_coef_).add(self.score_intercept_.values,axis="columns")
    
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
        # Set parallelize
        if self.parallelize:
            n_workers = -1
        else:
            n_workers = 1

        if not isinstance(X,pd.DataFrame):
            raise TypeError(
            f"{type(X)} is not supported. Please convert to a DataFrame with "
            "pd.DataFrame. For more information see: "
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        # Decision
        scores = self.decision_function(X)
        # Probabilité d'appartenance - transformation 
        C = mapply(mapply(scores,lambda x : np.exp(x),axis=0,progressbar=False,n_workers=n_workers),
                   lambda x : x/np.sum(x),axis=1,progressbar=False,n_workers=n_workers)
        return C

    def predict(self,X):
        """
        Predict class labels for samples in X
        -------------------------------------

        Parameters
        ----------
        X : DataFrame of shape (n_samples_, n_features_)
            The data matrix for which we want to get the predictions.
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples)
            Vectors containing the class labels for each sample
        """
        # check if X is an instance of polars dataframe
        if isinstance(X,pl.DataFrame):
            X = X.to_pandas()
        
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