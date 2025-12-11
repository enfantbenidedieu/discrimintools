# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin, ClassifierMixin, BaseEstimator
from numpy import exp, fill_diagonal, array, c_, nan, log, dot
from pandas import DataFrame, concat, merge
from scipy.special import softmax
from sklearn.metrics import classification_report, accuracy_score
from collections import namedtuple, OrderedDict

#intern functions
from .functions.utils import check_is_dataframe, check_is_series
from .functions.expand_grid import expand_grid

class _BaseDA(TransformerMixin,ClassifierMixin,BaseEstimator):
    """
    Base class for Discriminant Analysis methods

    .. warings::
        Don't use this class directly. Use the other classes of the package (CANDISC, DISCRIM, DiCA, LDAPC, MDA, CPLS, PLSDA & PLSLDA), which inherit from this Base class.
    """
    def __init__(self):
        pass

    def eval_predict(self,X,y,verbose=False):
        """
        Evaluation of the prediction' quality
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        y : Series of shape (n_samples,)
            True labels for ``X``.
        
        verbose : bool, default = False
            If True, print the infos.

        Returns
        -------
        NamedTuple

            obs : DataFrame of shape (1, 2)
                Observation Profile.
            cm : DataFrame of shape (n_classes + 1, n_classes + 1)
                Confusion matrix. Number of observations classified.
            resub : DataFrame of shape (n_classes + 1, n_classes + 1)
                Resubstitution Summary. Percent classified.
            error : DataFrame of shape (2, n_classes + 1)
                Error count.
            report : DataFrame of shape (n_targets + 3, 4)
                Classification report
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X=X)

        #set index name as None
        X.index.name = None

        #set number of samples
        n_read = X.shape[0]
        #drop rows with NA
        X = X.dropna()
        n_used = X.shape[0]
        #observations
        obs = DataFrame([[n_read,n_used]],columns=["Read","Used"],index=["Number of Observations"])
        
        #confusion matrix
        cm = self.pred_table(X,y)
        #percent classified
        resub = cm.copy()
        resub.loc["Total",:] = resub.sum(axis=0)
        resub = resub.apply(lambda x : x/x.sum(),axis=1).mul(100)
        resub.loc[:,"Total"] = resub.sum(axis=1)
        resub.loc["Priors",:] = [*self.call_.priors.values.tolist(),*[nan]]
        #copy of confusion matrix
        cm_c = cm.copy()
        #fill diagonal by 0
        fill_diagonal(cm_c.values,0)
        #error count estimates
        rate = cm_c.sum(axis=1)/cm.sum(axis=1)
        error = DataFrame(c_[array([rate,self.call_.priors.values]),[dot(rate,self.call_.priors.values.T),nan]],
                          index=["Rate","Priors"],columns= [*self.call_.classes,*["Total"]])
        #add total to table
        cm.loc["Total",:] = cm.sum(axis=0)
        cm.loc[:,"Total"] = cm.sum(axis=1)
        #classification report
        report = DataFrame(classification_report(y_true=y,y_pred=self.predict(X),labels=self.call_.classes,output_dict=True)).T
        #convert to dictionary
        eval_ = OrderedDict(obs=obs,cm=cm.astype(int),resub=resub,error=error,report=report)
        #convert to namedtuple
        eval_ = namedtuple("eval",eval_.keys())(*eval_.values())

        if verbose:
            print("Observation Profile:")
            print(obs)
            print("\nNumber of Observations Classified into {}:".format(self.call_.target))
            print(eval_.cm)
            print("\nPercent Classified into {}:".format(self.call_.target))
            print(eval_.resub)
            print("\nError Count Estimates for {}:".format(self.call_.target))
            print(eval_.error)
            print("\nClassification Report for {}:".format(self.call_.target))
            print(eval_.report)
        return eval_
      
    def pred_table(self,X,y):
        """
        Prediction table
        
        pred_table[i,j] refers to the number of times :math:`i` was observed and the model predicted :math:`j`. Correct predictions are along the diagonal.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            The data for which we want to get the prediction table.

        y : Series of shape (n_samples,)
            True labels for ``X``.

        Returns
        -------
        table : DataFrame of shape (n_classes, n_classes)
            The confusion matrix.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if y is an instance of class pd.Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_series(y)

        #set y name if None or an integer
        if y.name is None or isinstance(y.name, int):
            y.name = self.call_.target

        #grouping by target and prediction
        table = concat((y,self.predict(X)),axis=1).groupby(by=[self.call_.target,"prediction"],observed=False).size().to_frame("N").reset_index()
        #create all combinaison
        all_comb = expand_grid({self.call_.target : self.call_.classes, "prediction" : self.call_.classes})
        #left join and fill missing with zero
        table = (merge(all_comb,table,how="left",on=[self.call_.target,"prediction"]) #left join with all combinations
                    .assign(N = lambda x : x["N"].fillna(value=0).astype(int)) #fill NA with zero and convert to integer
                    .pivot(index=self.call_.target,columns="prediction",values="N") #convert to wider table
                    .loc[self.call_.classes, self.call_.classes]
                )
        return table
    
    def predict(self,X):
        """
        Predict class labels for samples in X

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            The data for which we want to get the predictions.
        
        Returns
        -------
        y_pred : Series of shape (n_samples,) 
            Predicted labels for ``X``.
        """
        y_pred = self.predict_proba(X).idxmax(axis=1)
        y_pred.name = "prediction"
        return y_pred
    
    def predict_log_proba(self,X):
        """
        Return log of posterior probabilities
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        C : DataFrame of shape (n_samples, n_classes)
            Posterior log-probabilities of classification per class.
        """
        scores = self.decision_function(X)
        log_likelihood = scores.sub(scores.max(axis=1),axis=0)
        return log_likelihood.sub(log(exp(log_likelihood).sum(axis=1)),axis=0)
    
    def predict_proba(self,X):
        """
        Estimate probability

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        Returns
        -------
        C : DataFrame of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        return DataFrame(softmax(self.decision_function(X),axis=1),columns=self.call_.classes,index=X.index)
    
    def score(self,X,y):
        """
        Return accuracy on the given input data

        Parameters
        -----------
        X : DataFrame of shape (n_samples, n_features)
            Input Data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        y : Series of shape (n_samples,)
            True labels for ``X``.

        Returns
        -------
        score : float 
            Mean accuracy of ``self.predict(X)`` w.r.t. ``y``.
        """
        return accuracy_score(y, self.predict(X))