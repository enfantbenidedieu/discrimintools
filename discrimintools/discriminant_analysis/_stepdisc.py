# -*- coding: utf-8 -*-
from numpy import linalg, exp, zeros
from pandas import DataFrame
from scipy import stats
from collections import OrderedDict, namedtuple
from sklearn.utils.validation import check_is_fitted

#intern functions
from ._base import _BaseDA
from ._discrim import DISCRIM
from ._candisc import CANDISC
from .functions.utils import check_is_dataframe
from .functions.splitmix import splitmix
from .functions.tab_disjunctive import tab_disjunctive

class STEPDISC(_BaseDA):
    """
    Stepwise Discriminant Analysis (STEPDISC)
    
    Given a classification variable and several quantitative variables, the :class:`~discrimintools.discriminant_analysis.STEPDISC` class performs a
    stepwise discriminant analysis to select a subset of the quantitative variables for use in discriminating among
    the classes. The set of variables that make up each class is assumed to be multivariate normal with a common
    covariance matrix. The :class:`~discrimintools.discriminant_analysis.STEPDISC` class can use forward selection and backward elimination, 
    which is a useful prelude to further analyses with the :class:`~discrimintools.discriminant_analysis.CANDISC` class or the :class:`~discrimintools.discriminant_analysis.DISCRIM` class.

    With :class:`~discrimintools.discriminant_analysis.STEPDISC`, variables are chosen to enter or leave the model according to the significance level of an `F` test from an analysis of covariance, 
    where the variables already chosen act as covariates and the variable under consideration is the dependent variable.
    Two selection methods are available: 'forward' and 'backward':

    1. Forward selection begins with no variables in the model. At each step, :class:`~discrimintools.discriminant_analysis.STEPDISC` enters the variable that contributes most to the discriminatory power of the model as measured by Wilks' lambda, the likelihood ratio criterion. When none of the unselected variables meet the entry criterion, the forward selection process stops.
    2. Backward elimination begins with all variables in the model except those that are linearly dependent on previous variables in the VAR statement. At each step, the variable that contributes least to the discriminatory power of the model as measured by Wilks' lambda is removed. When all remaining variables meet the criterion to stay in the model, the backward elimination process stops.

    Parameters
    ----------
    method : {'backward','forward'}, default='forward'
        The feature selection method to be used, possible values:
        - "forward" for forward selection, 
        - "backward" for backward elimination
    
    alpha : float, default = 1e-2
        The significance level for adding or retaining variables in stepwise variable selection.
    
    lambda_init : None or float, default = None
        Initial Wilks Lambda.
    
    verbose : bool, default=True 
        If `True`, print intermediary steps during feature selection (default)
    
    Returns
    -------
    call_ : NamedTuple
        Call informations:

        - obj : class
            An object of class CANDISC, DISCRIM
        - alpha : float
            The significance level for adding or retaining variables in stepwise variable selection.
        - target : str
            Name of target.
        - classes : list
            Names of classes
        - priors : Series of shape (n_classes,)
            Priors probabilities.

    disc_ : class
        An object of class CANDISC or DISCRIM

    model_ : str, default = "stepdisc"
        Name of model fitted.
        
    summary_ : NamedTuple
        Stepwise summary informations:
        
        - summary : DataFrame of shape (n_selected, 6)
            Summary of stepwise selection
        - selected : list
            Selected variables
        - removed : list
            Removed variables

    See also
    --------
    :class:`~discrimintools.discriminant_analysis.CANDISC`
        Canonical Discriminant Analysis (CANDISC)
    :class:`~discrimintools.discriminant_analysis.DISCRIM`
        Discriminant Analysis (linear and quadratic).
    :class:`~discrimintools.summary.summaryCANDISC`
        Printing summaries of Canonical Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryDISCRIM`
        Printing summaries of Discriminant Analysis (linear and quadratic) model.

    References
    ----------
    [1] Ricco Rakotomalala (2008), « `STEPDISC - Feature selection for LDA`_ », Université Lumière Lyon 2.

    [2] Ricco Rakotomalala (2012), « `Linear Discriminant Analysis - Tools comparison`_ », Université Lumière Lyon 2.

    [3] Ricco Rakotomalala (2014), « `Linear discriminant analysis (slides)`_ », Université Lumière Lyon 2.

    [4] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire`_ », Version 1.0, Université Lumière Lyon 2.

    [5] SAS/STAT 13.1 User's Guide (2013), « `The STEPDISC Procedure`_ », Chapter 93.

    .. _STEPDISC - Feature selection for LDA: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/en_Tanagra_Stepdisc.pdf
    .. _Linear Discriminant Analysis - Tools comparison: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/en_Tanagra_LDA_Comparisons.pdf
    .. _Linear discriminant analysis (slides): https://eric.univ-lyon2.fr/ricco/cours/slides/en/analyse_discriminante.pdf
    .. _Pratique de l'Analyse Discriminante Linéaire: https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf
    .. _The STEPDISC Procedure: https://support.sas.com/documentation/onlinedoc/stat/131/stepdisc.pdf

    Examples
    --------
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import DISCRIM, STEPDISC
    >>> D = load_heart("train") # load training data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = DISCRIM(method="linear")
    >>> clf.fit(X,y)
    >>> clf2 = STEPDISC(method="forward",alpha=0.01,verbose=True)
    >>> clf2.fit(clf)
    STEPDISC()
    """
    def __init__(
            self, method="forward", alpha=1e-2, lambda_init = None, verbose = True
    ):
        self.method = method
        self.alpha = alpha
        self.lambda_init = lambda_init
        self.verbose = verbose

    def decision_function(self,X):
        """
        Apply decision function to an input data

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            
        Returns
        -------
        C : DataFrame of shape (n_samples, n_classes)
            Decision function values related to each class, per sample.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #check if model has been updated
        if not hasattr(self,"disc_"):
            raise TypeError("\nSince only one feature is selected, {} procedure cannot be updated, therefore decision_function cannot be applied.".format(self.call_.obj.__class__.__name__))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X=X)

        #set index name as None
        X.index.name = None

        #check if X contains original features
        if not set(self.call_.obj.call_.Xtot.columns).issubset(X.columns):
            raise ValueError("The names of the features is not the same as the ones in the active features of the {} result".format(self.call_.obj.model_))
        
        #select original features
        X = X[self.call_.obj.call_.Xtot.columns]

        #split X
        split_X = splitmix(X)
        #extract elements
        X_quanti, X_quali, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.k1, split_X.k2

        #initialize DataFrame
        Xcod = DataFrame(index=X.index,columns=self.call_.obj.call_.X.columns).astype(float)

        #check if numerics variables
        if n_quanti > 0:
            #replace with numerics columns
            Xcod.loc[:,X_quanti.columns] = X_quanti
        
        #check if categorical variables      
        if n_quali > 0:
            #active categorics
            categorics = [x for x in self.call_.obj.call_.X.columns if x not in self.call_.obj.call_.Xtot.columns]
            #replace with dummies
            Xcod.loc[:,categorics] = tab_disjunctive(X=X_quali,dummies_cols=categorics,prefix=True,sep="")
        
        #remove non selected variables
        Xcod = Xcod.loc[:,list(self.call_.obj.cov_.total.index)] 
        return self.disc_.decision_function(Xcod)

    def fit(self,obj):
        """
        Fit Stepwise Discriminant Analysis procedure

        Parameters
        ----------
        obj : class
            An object of class CANDISC, DISCRIM

        Returns
        -------
        self : object
            Returns the instance itself        
        """
        # Wilks lambda
        def wilks(Vb, Wb) -> float:
            """
            Compute Wilk's Lambda

            Parameters
            ---------
            Vb : DataFrame of shape (n_features, n_features)
                Biaised total covariance matrix

            Wb : DataFrame of shape (n_features, n_features)
                Biaised pooled within-class covariance matrix

            Returns
            -------
            value : Wilk's Lambda
            """
            return linalg.det(Wb)/linalg.det(Vb)
        
        # Wilks lambda
        def wilks_log(Vb,Wb):
            """
            Compute Wilk's Lambda

            Parameters
            ----------
            Vb : DataFrame of shape (n_features, n_features)
                Biaised total covariance matrix

            Wb : DataFrame of shape (n_features, n_features)
                Biaised pooled within-class covariance matrix

            Returns
            -------
            value : Wilks Lambda
            """
            detVb, detWb = linalg.slogdet(Vb), linalg.slogdet(Wb)  
            # intra-classes biaisée
            return exp((detWb[0]*detWb[1])-(detVb[0]*detVb[1]))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if valid discriminan analysis model
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.model_ not in ["discrim", "candisc"]:
            raise TypeError("'model' must be an object of class DISCRIM, CANDISC")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if linear discriminant analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if obj.model_ == "discrim" and obj.method != "linear":
            raise TypeError("Only applied for linear discriminant analysis.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if alpha has a valid value
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.alpha is None:
            alpha = 1e-2
        elif not isinstance(self.alpha,float):
            raise TypeError("{} is not supported".format(type(self.alpha)))
        elif self.alpha < 0 or self.alpha > 1:
            raise ValueError("the 'alpha' value {} is not within the required range of 0 and 1.".format(self.alpha))
        else:
            alpha = self.alpha
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #initialize all elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #extract active elements : number of samples, number of features and number of classes
        n_samples, n_features, n_classes = obj.call_.n_samples, obj.call_.n_features, obj.call_.n_classes
        #extract total covariance and pooled withon covariance matrices
        tcovb, pwcovb = obj.cov_.btotal, obj.cov_.bpooled

        #initialize
        var_names, col_names = obj.call_.features, ["Wilks' Lambda","Partial R-Square","F Value","Num DF","Den DF","Pr>F"]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #forward procedure
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method.lower() == "forward":
            entered_var_list, entered_var_summary = [], []
            #wilk's lambda value for q = 0
            lw_init = 1
            #iteration
            for q in range(n_features):
                infos_vars = []
                for name in var_names:
                    #update entered var list
                    var_select = entered_var_list + [name]
                    # Wilks Lambda
                    lw = wilks_log(tcovb.loc[var_select, var_select],pwcovb.loc[var_select, var_select])
                    #degree of Freedom
                    ddl1, ddl2 = n_classes - 1, n_samples - n_classes - q
                    #fisher statistic
                    f_value = ddl2/ddl1 * ((lw_init/lw)-1)
                    #partial R squared
                    partiel_rsq = 1-(lw/lw_init)
                    #critical probability
                    p_value = stats.f.sf(f_value, ddl1, ddl2)
                    #concatenate
                    infos_vars.append((lw, partiel_rsq, f_value, ddl1, ddl2, p_value))

                #convert to DataFrame
                infos_step_results = DataFrame(infos_vars, index=var_names, columns=col_names)

                #to print step result
                if self.verbose:
                    print("\n====================== Step {} forward selection results =======================".format(q+1))
                    print(infos_step_results)

                #apply criteria decision
                entered_var = infos_step_results["Wilks' Lambda"].idxmin()
                #test
                if infos_step_results.loc[entered_var, "Pr>F"] > alpha:
                    if self.verbose:
                        print("\nNo variable can enter\n")
                    break
                else:
                    #print 
                    if self.verbose:
                        print("\nVariable {} will enter\n".format(entered_var))
                    entered_var_list.append(entered_var)
                    #update var names
                    var_names.remove(entered_var)
                    #update lw init
                    lw_init = infos_step_results.loc[entered_var,"Wilks' Lambda"]
                    #append summary
                    entered_var_summary.append(list(infos_step_results.loc[entered_var]))
                    
            stepdisc_summary = DataFrame(entered_var_summary, index=entered_var_list, columns=col_names)
            
            #selected variables
            selected_vars = list(stepdisc_summary.index)
            #removed variables
            removed_vars = [x for x in var_names if x not in selected_vars]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #backward procedure
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif self.method.lower() == "backward":
            removed_var_list, removed_var_summary = [], []
            #wilk's lambda value for pour q = p
            if self.lambda_init is None:
                lw_init = wilks(tcovb,pwcovb)
            else:
                lw_init = self.lambda_init
            
            for q in range(n_features, -1, -1):
                infos_vars = []
                for name in var_names:
                    #update lisrt of variables
                    var_select = [var for var in var_names if var != name]
                    #calcul du Lambda de Wilks (q-1)
                    lw = wilks_log(tcovb.loc[var_select,var_select],pwcovb.loc[var_select, var_select])
                    #degree of freedom
                    ddl1, ddl2 = n_classes - 1, n_samples - n_classes - q + 1
                    #fisher statistic
                    f_value = ddl2/ddl1*((lw/lw_init)-1)
                    #partial Rsquared
                    partiel_rsq = 1 - (lw_init/lw)
                    #critical probability
                    p_value = stats.f.sf(f_value, ddl1, ddl2)
                    #add to list
                    infos_vars.append((lw, partiel_rsq,f_value,ddl1,ddl2,p_value))

                #convert to DataFrame
                infos_step_results = DataFrame(infos_vars, index=var_names, columns=col_names)

                #po print step result
                if self.verbose:
                    print("\n====================== Step {} backward selection results =======================".format(n_features - q + 1))
                    print(infos_step_results)

                #apply criteria decision
                removed_var = infos_step_results["Wilks' Lambda"].idxmin()
                if infos_step_results.loc[removed_var, "Pr>F"] < alpha:
                    if self.verbose:
                        print("\nNo variable can be removed\n")
                    break
                else:
                    #print 
                    if self.verbose:
                        print("\nVariable {} will be removed\n".format(removed_var))

                    removed_var_list.append(removed_var)
                    #remove variables in global list
                    var_names.remove(removed_var)
                    #add
                    removed_var_summary.append(list(infos_step_results.loc[removed_var]))
                    #update init wilk's lambda
                    lw_init = infos_step_results.loc[removed_var,"Wilks' Lambda"]

            stepdisc_summary = DataFrame(removed_var_summary, index=removed_var_list, columns=col_names)

            #excluded variables
            removed_vars = list(stepdisc_summary.index)
            #selected variables
            selected_vars = [x for x in var_names if x not in removed_vars]
        else:
            raise ValueError("method should be one of 'backward', 'forward'.")
        
        #summary of stepwise discriminant analysis
        summary_ = OrderedDict(summary=stepdisc_summary,selected=selected_vars,removed=removed_vars)
        #convert to namedtuple
        self.summary_ = namedtuple("summary",summary_.keys())(*summary_.values())

        #initialize
        call_ = OrderedDict(obj=obj,alpha=alpha)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit model with selected features
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if len(selected_vars) > 1:
            #update X
            Xnew = obj.call_.X.loc[:,selected_vars]
            if obj.model_ == "discrim":
                self.disc_ = DISCRIM(method="linear",priors=obj.call_.priors,classes=obj.call_.classes,warn_message=False).fit(Xnew,obj.call_.y)
            elif obj.model_ == "candisc":
                self.disc_ = CANDISC(n_components=obj.call_.n_components,classes=obj.call_.classes,warn_message=False).fit(Xnew,obj.call_.y)
            
            #update call
            call_ = OrderedDict(**call_,**OrderedDict(target=self.disc_.call_.target,classes=self.disc_.call_.classes,priors=self.disc_.call_.priors))
        else:
            if self.verbose:
                print("\nSince only one feature is selected, {} procedure cannot be updated.".format(obj.__class__.__name__))
        
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())
               
        #set model name
        self.model_ = "stepdisc"
        return self
    
    def fit_transform(self,obj):
        """
        Fits transformer to ``X`` and returns a transformed version of samples.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features
        
        y : Series of shape (n_samples,)
            Target values. True labels for ``X``.
        
        Returns
        -------
        X_new : pandas DataFrame of shape (n_samples, n_components) or (n_samples, n_classes)
            Transformed samples, where ``n_components`` (resp. ``n_classes``) is the number of components (resp. classes).
        """
        self.fit(obj)
        #check if model has been updated
        if not hasattr(self,"disc_"):
            raise TypeError("\nSince only one feature is selected, {} procedure cannot be updated, therefore fit_transform cannot be applied.".format(obj.__class__.__name__))
        return self.transform(obj.call_.Xtot)
    
    def transform(self,X):
        """
        Project data to maximize class separation or dimensionality reduction 

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_components) or (n_samples, n_classes)
            Transformed data.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #check if model has been updated
        if not hasattr(self,"disc_"):
            raise TypeError("\nSince only one feature is selected, {} procedure cannot be updated, therefore fit_transform cannot be applied.".format(self.call_.obj.__class__.__name__))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X=X)

        #set index name as None
        X.index.name = None

        #check if X contains original features
        if not set(self.call_.obj.call_.Xtot.columns).issubset(X.columns):
            raise ValueError("The names of the features is not the same as the ones in the active features of the {} result".format(self.call_.obj.model_))
        
        #select original features
        X = X[self.call_.obj.call_.Xtot.columns]

        #split X
        split_X = splitmix(X)
        #extract elements
        X_quanti, X_quali, n_samples, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

        #initialize DataFrame
        Xcod = DataFrame(index=X.index,columns=self.call_.obj.call_.X.columns).astype(float)

        #check if numerics variables
        if n_quanti > 0:
            #replace with numerics columns
            Xcod.loc[:,X_quanti.columns] = X_quanti
        
        #check if categorical variables      
        if n_quali > 0:
            #active categorics
            categorics = [x for x in self.call_.obj.call_.X.columns if x not in self.call_.obj.call_.Xtot.columns]

            #replace with dummies
            Xcod.loc[:,categorics] = tab_disjunctive(X=X_quali,dummies_cols=categorics,prefix=True,sep="")
        
        #remove non selected variables
        Xcod = Xcod.loc[:,list(self.call_.obj.cov_.total.index)] 
        return self.disc_.transform(Xcod)