# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import pandas as pd
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin

from .lda import LDA
from .candisc import CANDISC

#######################################################################################################
#               Stepwise Discriminant Analysis (STEPDISC) - Discriminant Analysis Procedure
#######################################################################################################
class STEPDISC(BaseEstimator,TransformerMixin):
    """
    Stepwise Discriminant Analysis
    -------------------------------

    Performs a stepwise discriminant analysis to select a subset of the quantitative variables for use
    in discriminating among the classes. It can be used for forward selection, backward elimination, or
    stepwise selection.

    Parameters
    ----------
    level : float, default = 0.01
    method : {'forward','backward','stepwise'}, default = 'forward'
    alpha : the alpha level. The default alpha = 0.01.
    lambda_init : Initial Wilks Lambda/ Default = None
    verbose : 

    
    
    """
    

    def __init__(self,
                 method="forward",
                 alpha=0.01,
                 lambda_init = None,
                 model_train = False,
                 verbose = True,
                 parallelize=False):
        
        self.method = method
        self.alpha = alpha
        self.lambda_init = lambda_init
        self.model_train = model_train
        self.verbose = verbose
        self.parallelize = parallelize
    

    def fit(self,clf):
        """Fit

        Parameter
        ---------
        clf : an object of class LDA or CANDISC
        
        """

        if clf.model_ not in ["candisc","lda"]:
            raise TypeError("'clf' must be and object of class 'LDA' or 'CANDISC'.")
        
        isMethodValid = ["forward", "backward","stepwise"]
        if self.method.lower() not in isMethodValid:
            raise ValueError("'method' must be either 'backward','forward' or 'stepwise'.")

        self._compute_stats(clf)
        
        return self
    
    def _compute_forward(self,clf):
        raise NotImplementedError("Error : This method is not implemented yet.")
    
    def _compute_backward(self,clf):
        """
        Backward Elimination
        --------------------

        Parameters:
        -----------
        clf : an instance of class LDA or CANDISC
        
        Return
        ------
        resultat : DataFrame
        
        """
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
            return np.array([lwVar,lw/lwVar,fvalue,sp.stats.f.sf(fvalue, ddlsuppnum, ddlsuppden)])
        
        # Matrix V et W utilisées
        biased_V = ((clf.call_["X"].shape[0] - 1)/clf.call_["X"].shape[0])*clf.cov_["total"]
        biased_W = ((clf.call_["X"].shape[0] - clf.call_["n_classes"])/clf.call_["n_classes"])*clf.cov_["within"]

        # Lambda de Wilks - Initialisation de la valeur du Lambda de Wilks
        lambdaInit = 0.0
        if self.lambda_init is None:
            lambdaInit = np.linalg.det(biased_W)/np.linalg.det(biased_V)
        else:
            lambdaInit = self.lambda_init
        
        # Liste de variables pour le test
        listInit = clf.cov_["total"].index.tolist()

        # Sauvegarde des résultats
        Result = pd.DataFrame(columns=["Wilks L.","Partial L.","F","p-value"]).astype("float")
        listvarDel = list()

        # 
        while True:
            # Résultat d'un étape
            fextract = partial(fexclusion,W=biased_W,V=biased_V,n=clf.call_["X"].shape[0],K=clf.call_["n_classes"],lw=lambdaInit)
            res = pd.DataFrame(np.array(list(map(lambda j : fextract(j=j),np.arange(biased_W.shape[1])))),
                               columns=["Wilks L.","Partial L.","F","p-value"])
            res.index = listInit

            # Affichage : verbose == True
            if self.verbose:
                print(res)
                print()

            # Extraction de la ligne qui maximise la p-value
            id = np.argmax(res.iloc[:,3])
            
            if res.iloc[id,3] > self.alpha:
                # Nom de la variable à rétirer
                listvarDel.append(listInit[id])
                # Rajouter la ligne de résultats
                Result = pd.concat([Result,res.iloc[id,:].to_frame().T],axis=0)
                # Rétirer
                del listInit[id]
                #listInit.pop(id)

                if len(listInit) == 0:
                    break
                else:
                    # Retirer les lignes et les clonnes des matrices
                    biased_W = np.delete(biased_W.values,id,axis=0)
                    biased_W = np.delete(biased_W,id,axis=1)
                    biased_W = pd.DataFrame(biased_W,index=listInit,columns = listInit)

                    # Retirer les lignes et les clonnes des matrices
                    biased_V = np.delete(biased_V.values,id,axis=0)
                    biased_V = np.delete(biased_V,id,axis=1)
                    biased_V = pd.DataFrame(biased_V,index=listInit,columns = listInit)
                    # Mise à jour de Lambda Init
                    lambdaInit = res.iloc[id,0]
            else:
                break
        
        # Sauvegarde des résultats
        resultat = pd.DataFrame(Result,index=listvarDel,columns=["Wilks L.","Partial L.","F","p-value"])
        return resultat
        
    def _compute_stepwise(self,clf):
        raise NotImplementedError("Error : This method is not implemented yet.")

    def _compute_stats(self,clf):
        """
        
        
        """

        if self.method == "backward":
            overall_remove = self._compute_backward(clf=clf)
        elif self.method == "forward":
            overall_remove = self._compute_forward(clf=clf)
        elif self.method == "stepwise":
            overall_remove = self._compute_stepwise(clf=clf)

        features_remove = set(overall_remove.index)

        # Entraînement d'un modèle
        if self.model_train:
            # New data
            data = clf.call_["X"].drop(columns=features_remove)
            if clf.model_ == "lda":
                model = LDA(target=clf.call_["target"],priors=clf.call_["priors"]).fit(data)
                self.train_model_ = model
            elif clf.model_ == "candisc":
                model = CANDISC(target=clf.call_["target"],priors=clf.call_["priors"]).fit(data)
                self.train_model_ = model
        
        self.overall_remove_ = overall_remove
        self.features_remove_ = features_remove