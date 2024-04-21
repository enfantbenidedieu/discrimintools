# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin

from .lda import LDA
from .candisc import CANDISC

class STEPDISC(BaseEstimator,TransformerMixin):
    """
    Stepwise Discriminant Analysis (STEPDISC)
    -----------------------------------------

    Description
    -----------

    This class inherits from sklearn BaseEstimator and TransformerMixin class

    Performs a stepwise discriminant analysis to select a subset of the quantitative variables for use
    in discriminating among the classes. It can be used for forward selection, backward elimination.

    Parameters
    ----------
    model : an object of class LDA, CANDISC

    method : the feature selection method to be used :
                - "forward" for forward selection, 
                - "backward" for backward elimination

    alpha : Specifies the significance level for adding or retaining variables in stepwise variable selection, default = 0.01
            
    lambda_init : Initial Wilks Lambda/ Default = None

    model_train : if model should be train with selected variables

    verbose : boolean, 
                - if True, print intermediary steps during feature selection (default)
                - if False
    
    Return
    ------
    call_ : a dictionary with some statistics

    results_ : a dictionary with stepwise results

    model_ : string. The model fitted = 'stepdisc'

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    References
    ----------
    SAS documentation, https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.3/statug/statug_stepdisc_overview.htm
    Ricco Rakotomalala, Pratique de l'analyse discriminante linéaire, Version 1.0, 2020
    """
    def __init__(self,
                 model = None,
                 method="forward",
                 alpha=0.01,
                 lambda_init = None,
                 model_train = False,
                 verbose = True):
        
        # Compute Coariance and biased covariance matrix
        def cov_matrix(data):
            """
            Compute Covariance and biaised Covariance matrix
            ------------------------------------------------

            Parameters
            ----------
            data : pandas dataframe of shape (n_rows, n_features)

            Return
            ------
            a tuple of covariance and biaised covariance matrix            
            """
            n = data.shape[0] # numbeer of rows
            V = data.cov() # total covariance matrix
            Vb = ((n-1)/n) * V # biaised total covariance matrix
            return V, Vb
        
        # Compule pooled covariance matrix
        def pooled_cov_matrix(data,target):
            """
            Compute pooled covariance and biaised pooled covariance matrix
            --------------------------------------------------------------

            Parameters
            ----------
            data : pandas dataframe of shape (n_rows, n-features + 1)

            target : string, target name

            Return
            -----
            a tuple of pooled covariance and biaised pooled covariance matrix
            """
            n = data.shape[0] # number of rows
            classes = data[target].unique() # classes
            K = len(classes) # number of class
            V_k = data.groupby(target).cov(ddof=1)
            n_k = data[target].value_counts(normalize=False)

            # Pooled covariance matrix
            W = list(map(lambda k : (n_k[k]-1)*V_k.loc[k],classes))
            W = (1/(n-K))*reduce(lambda i,j : i + j, W)
            # Biaised pooled covariance matrix
            Wb = ((n-K)/n)*W
            return W, Wb
        
        # Wilks lambda
        def wilks(Vb, Wb):
            """
            Compute Wilks Lambda
            --------------------

            Parameter
            ---------
            Vb : biaised total covariance matrix

            Wb : biaised pooled covariance matrix

            Return
            ------
            value : Wilks Lambda
            """
            return np.linalg.det(Wb)/np.linalg.det(Vb)
        
        # Wilks lambda
        def wilks_log(Vb, Wb):
            """
            Compute Wilks Lambda
            --------------------

            Parameter
            ---------
            Vb : biaised total covariance matrix

            Wb : biaised pooled covariance matrix

            Return
            ------
            value : Wilks Lambda
            """
            detVb = np.linalg.slogdet(Vb)  
            detWb = np.linalg.slogdet(Wb)  
            # intra-classes biaisée
            return np.exp((detWb[0]*detWb[1])-(detVb[0]*detVb[1]))

        # Critical probability
        def p_value(F, ddl1, ddl2):
            """
            Compute critical probability
            ----------------------------

            Parameters
            ----------
            F : Fisher statistic

            ddl1 : first degree of freedom

            ddl2 : second degree of freedom

            Return
            ------
            pvalue : float, critical probability
            """
            if (F < 1):
                return (1.0 - sp.stats.f.cdf(1.0/F, ddl1, ddl2))
            return (1.0 - sp.stats.f.cdf(F, ddl1, ddl2))

        # Check if valid discriminan analysis model
        if model.model_ not in ["lda", "candisc"]:
            raise TypeError("'model' must be an object of class LDA, CANDISC")
        
        # Chack if valid method
        isMethodValid = ["forward", "backward"]
        if method.lower() not in isMethodValid:
            raise ValueError("'method' should be one of 'backward','forward'")
        
        ####### Global dataset
        dataset = model.call_["X"]
        # Number of rows
        n = dataset.shape[0]

        ###### Matrice of features
        X = model.call_["X"][model.call_["features"]]
        # Number of features
        p = model.call_["n_features"]

        # Vetor of target
        y = dataset[[model.call_["target"]]]
        # Number of classes
        K = len(np.unique(y).tolist())

        #################################################################################"
        _, Vb = cov_matrix(X)
        _, Wb = pooled_cov_matrix(data=dataset, target=model.call_["target"])
        
        #####
        varNames = X.columns.tolist()
        colNames = ["Wilks L.","Partial L.","F","p-value"]

        ########################## Forward procedure
        if method.lower() == "forward":
            enteredVarList = []
            enteredVarSummary = []
            L_initial = 1  # Valeur du Lambda de Wilks pour q = 0
            for q in range(p):
                infoVar = []
                for name in varNames:
                    # Wilks Lambda
                    wilksVarSelect = enteredVarList+[name]
                    L = wilks_log(Vb.loc[wilksVarSelect, wilksVarSelect],Wb.loc[wilksVarSelect, wilksVarSelect])
                    # Degree of Freedom
                    ddl1, ddl2 = K-1, n-K-q
                    # Fisher statistic
                    F = ddl2/ddl1 * (L_initial/L-1)
                    R = 1-(L/L_initial)  # Calcul du R² partiel
                    # Critical probability
                    pval = p_value(F, ddl1, ddl2)
                    infoVar.append((L, R, F, pval))

                infoStepResults = pd.DataFrame(infoVar, index=varNames, columns=colNames)

                # To print step result
                if verbose:
                    print(infoStepResults)
                    print()

                # Apply criteria decision
                enteredVar = infoStepResults["Wilks L."].idxmin()
                if infoStepResults.loc[enteredVar, "p-value"] > alpha:
                    break
                else:
                    enteredVarList.append(enteredVar)
                    varNames.remove(enteredVar)
                    L_initial = infoStepResults.loc[enteredVar,"Wilks L."]
                    enteredVarSummary.append(list(infoStepResults.loc[enteredVar]))
                    
            stepdiscSummary = pd.DataFrame(enteredVarSummary, index=enteredVarList, columns=colNames)
            ##########
            # Selected variables
            selectedVar = stepdiscSummary.index.tolist()
            # Removed variables
            removedVar = [x for x in varNames if x not in selectedVar]
        elif method.lower() == "backward":
            removedVarList = []
            removedVarSummary = []
            # Calcul du Lamba de Wilks pour q = p
            if lambda_init is None:
                L_initial = wilks(Vb,Wb)
            else:
                L_initial = lambda_init
            for q in range(p, -1, -1):
                infoVar = []
                for name in varNames:
                    # calcul du Lambda de Wilks (q-1)
                    wilksVarSelect = [var for var in varNames if var != name]
                    L = wilks_log(Vb.loc[wilksVarSelect, wilksVarSelect],Wb.loc[wilksVarSelect, wilksVarSelect])
                    # Degree of freedom
                    ddl1, ddl2 = K-1, n-K-q+1
                    # Fisher statistic
                    F = ddl2/ddl1*(L/L_initial-1)
                    # Partial Rsquared
                    R = 1-(L_initial/L)
                    # Critical probability
                    pval = p_value(F, ddl1, ddl2)
                    infoVar.append((L, R, F, pval))

                infoStepResults = pd.DataFrame(infoVar, index=varNames, columns=colNames)

                # To print step result
                if verbose:
                    print(infoStepResults)
                    print()

                # Apply criteria decision
                removedVar = infoStepResults["Wilks L."].idxmin()
                if infoStepResults.loc[removedVar, "p-value"] < alpha:
                    break
                else:
                    removedVarList.append(removedVar)
                    varNames.remove(removedVar)
                    removedVarSummary.append(list(infoStepResults.loc[removedVar]))
                    L_initial = infoStepResults.loc[removedVar,"Wilks L."]

            stepdiscSummary = pd.DataFrame(removedVarSummary, index=removedVarList, columns=colNames)

            ###### Selected variables
            removedVar = stepdiscSummary.index.tolist()
            selectedVar = [x for x in varNames if x not in removedVar]

        # 
        self.call_ = {"model" : model, "method" : method, "alpha" : alpha}
        # Results
        self.results_ = {"summary" : stepdiscSummary,"selected" : selectedVar,"removed" : removedVar}

        ##################################### Train model with selected variables
        if model_train:
            # New data
            if model.model_ == "lda":
                new_model = LDA(target=[model.call_["target"]],features=selectedVar,priors=model.call_["priors"]).fit(dataset)
            elif model.model_ == "candisc":
                new_model = CANDISC(target=[model.call_["target"]],priors=model.call_["priors"]).fit(dataset)
            
            self.results_ = {**self.results_,**{"train" : new_model}}
        
        self.model_ = "stepdisc"