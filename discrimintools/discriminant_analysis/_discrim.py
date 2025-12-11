# -*- coding: utf-8 -*-
from numpy import ndarray, array, c_, diag, dot, linalg, log, sqrt, average, ones
from pandas import DataFrame, concat, CategoricalDtype, Series
from pandas.api.types import is_string_dtype
from functools import reduce
from collections import OrderedDict, namedtuple
from statsmodels.multivariate.manova import MANOVA
from sklearn.utils.validation import check_is_fitted

#intern function
from ._base import _BaseDA
from .functions.utils import check_is_dataframe, check_is_series, check_is_bool
from .functions.preprocessing import preprocessing
from .functions.model_matrix import model_matrix
from .functions.ldavip import ldavip
from .functions.describe import describe
from .functions.sscp import sscp
from .functions.box_m_test import box_m_test
from .functions.cov_to_cor_test import cov_to_cor_test
from .functions.distance import sqmahalanobis
from .functions.cov_infos import cov_infos
from .functions.univ_test import univ_test
from .functions.diagnostics import diagnostics
from .functions.splitmix import splitmix
from .functions.tab_disjunctive import tab_disjunctive

class DISCRIM(_BaseDA):
    """
    Discriminant Analysis (DISCRIM)
    
    Performs a discriminant analysis (linear and quadratic) on a set of observations (training data) containing one or more numerics variables and a classification variables defining groups of observations.
    The derived discriminant criterion from the training data can be applied to a testing dataset.

    Parameters
    ----------
    method : {'linear', 'quad'}, default = 'linear'
        The discriminant analysis method to performs, possible values: 

        - 'linear' for linear discriminant analysis (LDA).
        - 'quad' for quadratic discriminant analysis (QDA)
    
    priors : str or array-like or Series of shape (n_classes,), default = None
        The priors statement specifies the class prior probabilities of group membership, possibles values: 

        - 'equal' to set the prior probabilities equal.
        - 'prop' to set the prior probabilities proportional to the sample sizes.
        - numpy 1-D array or Series which specify the prior probability for each level of the classification variable.
    
    classes : None, tuple or list, default = None
        Name of level in order to return. If None, classes are sorted in unique values in y.
    
    var_select : bool, default = False
        Whether to applied feature selection based on variable importance (contribution) in prediction for linear discriminant analysis

    level : float, default = None
        Significance level for the variable importance critical probability. You can specify the `level` option only when both method = 'linear' and var_select=True are also specified.
        If you specify both method = 'linear' and var_select=True but omit the `level` option, DISCRIM uses :math:`5e-2` as the significance level for the variabe importance.
    
    tol : float, default = None
        Significance level for the test of homogeneity. You can specify the `tol` option only when method = 'quad' is also specified.
        If you specify method = 'quad' but omit the `tol` option, DISCRIM uses :math:`1e-1` as the significance level for the test.
    
    warn_message : bool, default = True
        Show warning messages. Raise a warning without making the program crash.

    Returns
    -------
    call_ :  NamedTuple
        Call informations:

        - Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.
        - X : DataFrame of shape (n_samples, n_features)
            Training data.
        - y : Series of shape (n_samples,)
            Target values. True values for ``X``.
        - target : str
            Name of target.
        - features : list
            Names of features seen during ``fit``.
        - classes : list
            Names of classes.
        - priors : Series of shape (n_classes,)
            Priors probabilities.
        - n_samples : int
            Number of samples.
        - n_features : int
            Number of features.
        - n_classes : int
            Number of target values.

    classes_ : Namedtuple 
        Classes informations:

        - infos : DataFrame of shape (n_classes, 3)
            class level information (frequency, proportion, prior probability).
        - center : DataFrame of shape (n_classes, n_features) 
            Class means.
        - total : DataFrame of shape (n_features, n_classes)
            Total-sample standardized class means.
        - pooled : DataFrame of shape (n_features, n_classes)
            Pooled-within class standardized class means.
        - mahal : DataFrame of shape (n_classes, n_classes)
            Squared Mahalanobis distances between classes.
        - gen : DataFrame of shape (n_classes, n_classes)
            Generalized Squared distances between classes.

    coef_ : DataFrame of shape (n_features, n_classes)
        Linear classification functions coefficients.

    corr_ : NamedTuple 
        Correlation coefficients test:

        - total : DataFrame of shape (C^{2}_{n_features}, 7)
            Total-sample correlation coefficients test.
        - within : dict 
            Within-class correlation coefficients test.
        - pooled : DataFrame of shape (C^{2}_{n_features}, 7)
            Pooled within-class correlation coefficients test.
        - between : DataFrame of shape (C^{2}_{n_features}, 7)
            Between-class correlation coefficients test.

    cov_ : NamedTuple
        Covariance matrices:

        - total : DataFrame of shape (n_features, n_features)
            Total-sample covariance matrix.
        * btotal : DataFrame of shape (n_features, n_features)
            Biased total-sample covariance matrix.
        - within : dict 
            Within-class covariance matrices.
        - bwithin : dict
            Biased within-class covariance matrices.
        - pooled : DataFrame of shape (n_features, n_features)
            pooled within-class covariance matrix.
        - bpooled : DataFrame of shape (n_features, n_features)
            biased pooled within-class covariance matrix.
        - between : DataFrame of shape (n_features, n_features)
            Between-class covariance matrix
        - bbetween : DataFrale of shape (n_features, n_features)
            biased between-class covariance matrix.
        - test : DataFrame of shape (1, 7)
            Box's M test.

    ind_ : NamedTuple
        Individuals informations:

        - scores : DataFrame of shape (n_samples, n_classes) 
            The total scores of individuals.
        - mahal : DataFrame of shape (n_samples, n_classes) 
            Squared Mahalanobis distance to origin.
        * gen : DataFrame shape (n_samples, n_classes) 
            Generalized squared distance to origin.

    model_ : str, default = "discrim"
        Name of model fitted.

    sscp_ : NamedTuple 
        Sum of square cross product (SSCP) matrices:

        - total : DataFrame of shape (n_features, n_features)
            Total-sample SSCP matrix
        - within : dict
            Within-class SSCP matrices
        - pooled: DataFrame of shape (n_features, n_features)
            Pooled within-class SSCP matrix
        - between : DataFrame of shape (n_features, n_features)
            Between-class SSCP matrix.

    statistics_ : NamedTuple
        Statistics results:

        - anova : DataFrame of shape (n_features, 11)
            Analysis of variance test.
        - manova : DataFrame of shape (4, 5)
            Multivariate analysis of variance test.
        - average_rsq : DataFrame of shape (1, 2)
            Average R-square.
        - performance : DataFrame of shape (3, 3)
            The model global performance. Only if linear discriminant analysis.

    summary_ : NamedTuple
        Summary informations:

        - infos : DataFrame of shape (3, 4)
            Summary informations (total sample size, number of features, number of classes, 
            total degree of freedom, within-class degree of freedom, between-class degree of freedom).
        - total : DataFrame of shape (n_features, 8)
            Total-sample statistics, see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
        - within : dict
            Within-class statistics.
    
    vip_ : NamedTuple
        Variable importance for prediction:

        - vip : DataFrame of shape (n_features, 6)
            Variable importance for prediction.
        - selected : list
            Selected variables.
    
    See also
    --------
    :class:`~discrimintools.discriminant_analysis.GFALDA`
        General Factor Analysis Linear Discriminant Analysis
    :class:`~discrimintools.discriminant_analysis.CPLS`
        Partial Least Squares for Classification
    :class:`~discrimintools.discriminant_analysis.PLSDA`
        Partial Least Squares Discriminant Analysis
    :class:`~discrimintools.discriminant_analysis.PLSLDA`
        Partial Least Squares Linear Discriminant Analysis
    :class:`~discrimintools.summary.summaryDISCRIM`
        Printing summaries of Discriminant Analysis (linear & quadratic) model.
    :class:`~discrimintools.summary.summaryDA`
        Printing summaries of Discriminant Analysis model.
        
    References
    ----------
    [1] Bardos M. (2001), « Analyse discriminante - Application au risque et scoring financier », Dunod.
    
    [2] Lebart Ludovic, Piron Marie, & Morineau Alain (2006), « `Statistique Exploratoire Multidimensionnelle <https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-12/010038111.pdf>`_ », Dunod, Paris 4ed.
    
    [3] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_ », Université Lumière Lyon 2, Version 1.0.
    
    [4] Saporta Gilbert (2011), « `Probabilités, Analyse des données et Statistiques <https://en.pdfdrive.to/dl/probabilites-analyses-des-donnees-et-statistiques>`_ », Editions TECHNIP, 3ed.
    
    [5] Tenenhaus Michel (1996), « Méthodes statistiques en gestion », Dunod.
    
    [6] Tuffery Stephane (2017), « Data Mining et Statistique décisionelle », Editions TECHNIP, 5ed.
    
    [7] Tuffery Stephane (2025), « Data Science, Statistique et Machine learning », Editions TECHNIP, 6ed.

    [8] SAS/STAT 13.2 User's Guide (2014), « `The DISCRIM Procedure <https://support.sas.com/documentation/onlinedoc/stat/132/discrim.pdf>`_ », Chapter 35.

    Examples
    --------
    >>> from discrimintools.datasets import load_alcools
    >>> from discrimintools import DISCRIM
    >>> D = load_alcools() # load training data
    >>> y, X = D["TYPE"], D.drop(columns["TYPE"]) # split into X and y
    >>> #linear discriminant analysis (LDA)
    >>> clf = DISCRIM()
    >>> clf.fit(X,y)
    DISCRIM(priors='prop')
    >>> #quadratic discriminant analysis
    >>> clf2 = DISCRIM(method='quad')
    >>> clf2.fit(X,y)
    DISCRIM(method='quad',priors='prop')
    ```
    """
    def __init__(
            self, method = 'linear', priors = None, classes = None, var_select = False, level = None, tol = None, warn_message = True
    ):
        self.method = method 
        self.priors = priors
        self.classes = classes
        self.var_select = var_select
        self.level = level
        self.tol = tol
        self.warn_message = warn_message

    def decision_function(self,X):
        """
        Apply decision function to an input data

        The decision function is equal to the `log-posterior`_ of the model.

        .. log-posterior: https://online.stat.psu.edu/stat857/node/80/

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

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X=X)

        #set index name as None
        X.index.name = None

        #check if X contains original features
        if not set(self.call_.Xtot.columns).issubset(X.columns):
            raise ValueError("The names of the features is not the same as the ones in the active features of the DISCRIM result")
        #select original features
        X = X[self.call_.Xtot.columns]

        #split X
        split_X = splitmix(X)
        #extract elements
        X_quanti, X_quali, n_samples, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

        #initialize DataFrame
        Xcod = DataFrame(index=X.index,columns=self.call_.X.columns).astype(float)

        #check if numerics variables
        if n_quanti > 0:
            #replace with numerics columns
            Xcod.loc[:,X_quanti.columns] = X_quanti
        
        #check if categorical variables      
        if n_quali > 0:
            #active categorics
            categorics = [x for x in self.call_.X.columns if x not in self.call_.Xtot.columns]
            #replace with dummies
            Xcod.loc[:,categorics] = tab_disjunctive(X=X_quali,dummies_cols=categorics,prefix=True,sep="")

        #remove non selected variables
        Xcod = Xcod.loc[:,list(self.cov_.total.index)] 

        #chi-square pvalue and tolerance threshold
        p_value, tol = self.cov_.test.iloc[0,6], self.call_.tol
        #quadratic discriminant analysis
        if self.method == "quad" and  p_value <= tol:
            #inverse of within-class covariance matrix
            invwcov = {k : linalg.inv(self.cov_.within[k]) for k in self.call_.classes}
            #squared distance of individuals to origin
            mahal = concat((sqmahalanobis(X=Xcod,VI=invwcov[k],mu=self.classes_.center.loc[k,:]).to_frame(k) for k in self.call_.classes),axis=1)
            #generalized squared distance of individuals to origin
            gsqdist = mahal.add(self.cov_.infos.loc[self.call_.classes,"Natural Log of the Determinant"],axis=1) 
        #linear discriminant analysis
        elif (self.method == "linear") or (self.method == "quad" and  p_value > tol):
            #inverse of pooled within-class covariance matrix
            invpwcov = linalg.inv(self.cov_.pooled)
            #generalized squared to distance of individuals to origin
            gsqdist = concat((sqmahalanobis(X=Xcod,VI=invpwcov,mu=self.classes_.center.loc[k,:]).to_frame(k) for k in self.call_.classes),axis=1)
        
        #remove priors log-probabilities
        if not (isinstance(self.priors, str) and self.priors == "equal"):
            gsqdist = gsqdist.sub(2*log(self.call_.priors),axis=1)
        return -0.5*gsqdist
    
    def feature_importance(self,level=5e-2,all_vars=True):
        """
        Variables Importance for Prediction in Linear Discriminant Analysis (LDAVIP)

        Parameters
        ----------
        level : float, default=5e-2
            Significance level for the variable importance critical probability.
            If None :math:`5e-2` is used as the significance level for the variabe importance.

        all_vars : bool, default=True
            If to test all subset of variables.

        Returns
        ------- 
        vip : DataFrame of shape (n_features, 6)
            Variable importance for prediction.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if linear discriminant analysis model
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method != "linear":
            raise NotImplementedError("'feature_importance' method cannot be used for QDA method.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if var_select is False
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.var_select:
            raise  NotImplementedError("'feature_importance' method cannot be used if var_select=True.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if level is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if level is None:
            level = 5e-2
        elif not isinstance(level,float):
            raise TypeError("{} is not supported".format(type(level)))
        elif level < 0 or level > 1:
            raise ValueError("the 'level' value {} is not within the required range of 0 and 1.".format(level))
        else:
            level = level
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if boolean
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(all_vars)
        
        return ldavip(X=self.call_.X,y=self.call_.y,level=level,all_vars=all_vars).vip
    
    def fit(self,X,y):
        """
        Fit the Discriminant Analysis model.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training Data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        y : Series of shape (n_samples,)
            Target values. True labels for ``X``.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if y is an instance of class pd.Series
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_series(y)

        #check if len are equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X must be equal to the number of samples in y")

        #check if all elements in y are string
        if not all(isinstance(kq, str) for kq in y):
            raise TypeError("All elements in y must be a string")
        
        #set y name if None
        if y.name is None or isinstance(y.name, int):
            y.name = "group"
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if method is 'linear' or 'quad'
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method not in ['linear','quad']:
            raise ValueError("method must be one of 'linear', 'quad', got {}".format(self.method))
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if priors is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.priors is None:
            self.priors = "prop"
        elif not isinstance(self.priors,(str,list,tuple,ndarray,Series)):
            raise TypeError("{} is not supported".format(type(self.priors)))
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if var_select is a bool
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.var_select)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if level is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method == 'linear':
            if self.level is None:
                level = 5e-2
            elif not isinstance(self.level,float):
                raise TypeError("{} is not supported".format(type(self.level)))
            elif self.level < 0 or self.level > 1:
                raise ValueError("the 'level' value {} is not within the required range of 0 and 1.".format(self.level))
            else:
                level = self.level
        else:
            level = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if tol is not None (for quadratic discriminant analysis)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method == 'quad':
            if self.tol is None:
                tol = 1e-1
            elif not isinstance(self.tol,float):
                raise TypeError("{} is not supported".format(type(self.tol)))
            elif self.tol < 0 or self.tol > 1:
                raise ValueError("the 'tol' value {} is not within the required range of 0 and 1.".format(self.tol))
            else:
                tol = self.tol
        else:
            tol = None
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if warn_message is a bool
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.warn_message)

        #make a copy of original data
        Xtot = X.copy(deep=True)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)
         
        #warning message to inform
        if self.warn_message:
            if any(is_string_dtype(X[k]) for k in X.columns):
                print("\nCategorical features have been encoded into binary variables.\n")

        #encode categorical variables into binary without first level.
        X = model_matrix(X=X)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set classes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #unique element in y
        uq_y = sorted(list(y.unique()))
        #number of classes
        n_classes = len(uq_y)
        if self.classes is not None and isinstance(self.classes, (list,tuple)):
            if len(list(set(self.classes) & set(uq_y))) != n_classes:
                raise ValueError("Insert good classes")
            classes = [str(k) for k in self.classes]
        else:
            classes = uq_y

        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=classes,ordered=True))
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables importance for prediction in linear discriminant analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.method == 'linear':
            vip = ldavip(X,y,level,all_vars=False)
            #check if at least one variable selected
            if self.var_select and len(vip.selected) > 0:
                #update X with selected variables
                X = X.loc[:,vip.selected]
                #update variable importance
                vip = ldavip(X,y,level,all_vars=False)
            #set to attribute
            self.vip_ = vip

        #number of samples and number of features
        n_samples, n_features = X.shape
        #set target and features names
        target, features = y.name, X.columns.tolist()

        #define subset of X
        X_k = {k : X.loc[y[y==k].index,:] for k in classes}
        #count and proportion
        n_k, p_k = y.value_counts(normalize=False).loc[classes],  y.value_counts(normalize=True).loc[classes]

        #set piors probabilities
        if isinstance(self.priors,str):
            if self.priors == "prop":
                priors = array(p_k)
            elif self.priors == "equal":
                priors = ones(n_classes)/n_classes
            else:
                raise TypeError("Specify a right value for piors")
        elif isinstance(self.priors,(list,tuple,ndarray,Series)):
            priors = array([x/sum(self.priors) for x in self.priors])

        #check if any value in priors is negative
        if any(x < 0 for x in priors):
            raise ValueError("priors must be non-negative")

        #convert to pandas Series
        priors = Series(priors,index=classes,name="priors")

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=Xtot,X=X,y=y,target=target,features=features,classes=classes,priors=priors,n_samples=n_samples,n_features=n_features,n_classes=n_classes,level=level,tol=tol)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #sample statistics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #summary information
        summary_infos = DataFrame({"Infos" : ["Total Sample Size","Variables","Classes"],
                                   "Value" : [n_samples,n_features,n_classes],
                                   "DF" : ["DF Total", "DF Within Classes", "DF Between Classes"],
                                   "DF value" : [n_samples - 1, n_samples - n_classes, n_classes - 1]})
        #total-sample and within-class summaries
        tsummary, wsummary = describe(X), {k : describe(X_k[k]) for k in classes}
        #convert to dictionary
        summary_ = OrderedDict(infos=summary_infos,total=tsummary,within=wsummary)
        #convert to namedtuple
        self.summary_ = namedtuple("summary",summary_.keys())(*summary_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #sum of square cross product (SSCP) matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample and within-class SSCP matrix
        tsscp, wsscp = sscp(X=X), {k: sscp(X_k[k]) for k in classes}
        #pooled within-class SSCCP matrix
        pwsscp = reduce(lambda i , j : i + j, wsscp.values())
        #between-class SSCP matrix
        bsscp = tsscp - pwsscp
        #convert to dictionary
        sscp_ = OrderedDict(total=tsscp,within=wsscp,pooled=pwsscp,between=bsscp)
        #convert to namedtuple
        self.sscp_ = namedtuple("sscp",sscp_.keys())(*sscp_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #covariance matrices
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample and biased total-sample covariance matrices
        tcov, tcovb = tsscp.div(n_samples - 1), tsscp.div(n_samples)
        #within-class and biased within-class covariance matrices
        wcov, wcovb = {k : wsscp[k].div(n_k[k]-1) for k in classes}, {k : wsscp[k].div(n_k[k]) for k in classes}
        #test of equality of covariance matrix - homogeneity of variance and covariance
        wcov_test = box_m_test(wcov.values(),list(n_k))
        #chi-square pvalue
        p_value = wcov_test.iloc[0,6]
        #print warning message
        if self.warn_message:
            if self.method == "quad" and  p_value > tol:
                print("\nSince the Chi-Square value is not significant at the {} level, a pooled covariance matrix will be used in the discriminant function.\nReference: Morrison, D.F. (1976) Multivariate Statistical Methods p252.".format(tol))
            elif self.method == "quad" and  p_value <= tol:
                print("\nSince the Chi-Square value is significant at the {} level, the within covariance matrices will be used in the discriminant function.\nReference: Morrison, D.F. (1976) Multivariate Statistical Methods p252.".format(tol))

        #pooled within-class and biased pooled within-class covariance matrices
        pwcov, pwcovb = pwsscp.div(n_samples - n_classes), pwsscp.div(n_samples)
        #between-class and biased between-class covariance matrices
        bcov, bcovb  = bsscp.div(n_samples*(n_classes-1)/n_classes), bsscp.div(n_samples)
        ##covariance matrices informations - rank and natural log of the determinant
        cov_info = cov_infos(X=pwcov).to_frame("Pooled").T
        if self.method == "quad":
            #within-class covariance matrices informations
            wcov_infos = concat((cov_infos(X=wcov[k]).to_frame(k) for k in classes),axis=1).T
            #concatenate
            cov_info = concat((cov_info,wcov_infos),axis=0)
        #convert to integer
        cov_info["Rank"] = cov_info["Rank"].astype(int)
        #convert to dictionary
        cov_ = OrderedDict(infos=cov_info,total=tcov,btotal=tcovb,within=wcov,bwithin=wcovb,pooled=pwcov,bpooled=pwcovb,between=bcov,bbetween=bcovb,test=wcov_test)
        #convert to namedtuple
        self.cov_ = namedtuple("cov",cov_.keys())(*cov_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlation coefficients test
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total sample and within-class correlation coefficients
        tcortest, wcortest  = cov_to_cor_test(X=tcovb,n_samples=n_samples), {k: cov_to_cor_test(X=wcovb[k],n_samples=n_k[k]) for k in classes}
        #pooled within-class and between-class correlation coefficients
        pwcortest, bcortest = cov_to_cor_test(X=pwcov,n_samples=n_samples-n_classes+1), cov_to_cor_test(X=bcov,n_samples=n_classes)
        #convert to dictionary
        cortest_ = OrderedDict(total=tcortest,within=wcortest,pooled=pwcortest,between=bcortest)
        #convert to namedtuple
        self.corr_ = namedtuple("corr",cortest_.keys())(*cortest_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classes and individuals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #class level information
        class_infos = DataFrame(c_[n_k,p_k,priors],columns=["Frequency","Proportion","Prior Probability"],index=classes)
        class_infos["Frequency"] = class_infos["Frequency"].astype(int)
        #within-class average
        wcenter = concat((X_k[k].mean(axis=0).to_frame(k) for k in classes),axis=1).T
        #total-sample standardized class means
        tcenter = wcenter.sub(tsummary["mean"],axis=1).div(tsummary["std"],axis=1).T
        #pooled within-class standardized class means
        pcenter = wcenter.sub(tsummary["mean"],axis=1).div(sqrt(diag(pwcov)),axis=1).T
        #convert to dictionary
        classes_, ind_ = OrderedDict(infos=class_infos,center=wcenter,total=tcenter,pooled=pcenter), OrderedDict()
        
        #squared distance and generalized squared distance
        if self.method == "quad" and  p_value <= tol:
            #inverse of within-class covariance matrices
            invwcov = {k : linalg.inv(wcov[k]) for k in classes}
            #squared distance of classes to origin
            class_mahal = concat((sqmahalanobis(X=wcenter,VI=invwcov[k],mu=wcenter.loc[k,:]).to_frame(k) for k in classes),axis=1)
            #generalized squared distance of classes to origin
            class_gen = class_mahal.add(wcov_infos.loc[classes,"Natural Log of the Determinant"],axis=1) 

            #squared distance of individuals to origin
            ind_mahal = concat((sqmahalanobis(X=X,VI=invwcov[k],mu=wcenter.loc[k,:]).to_frame(k) for k in classes),axis=1)
            #generalized squared distance of individuals to origin
            ind_gen = ind_mahal.add(wcov_infos.loc[classes,"Natural Log of the Determinant"],axis=1) 
        elif (self.method == "linear") or (self.method == "quad" and  p_value > tol):
            #inverse pooled within-class covariance matrix
            invpwcov = linalg.inv(pwcov)
            #squared distance of classes to origin
            class_mahal = concat((sqmahalanobis(X=wcenter,VI=invpwcov,mu=wcenter.loc[k,:]).to_frame(k) for k in classes),axis=1)
            #generalized squared distance of classes to origin
            class_gen = class_mahal.copy()

            #squared distance of individuals to origin
            ind_mahal = concat((sqmahalanobis(X=X,VI=invpwcov,mu=wcenter.loc[k,:]).to_frame(k) for k in classes),axis=1)
            #generalized squared to distance of individuals to origin
            ind_gen = ind_mahal.copy()

            #coefficients of features
            coef = DataFrame(dot(invpwcov,wcenter.T),index=X.columns,columns=classes)
            #intercept
            intercept = - 0.5*diag(dot(dot(wcenter,invpwcov),wcenter.T))
            if not (isinstance(self.priors,str) and self.priors == "equal"):
                intercept = intercept + log(priors)
            #convert to DataFrame
            intercept = Series(intercept,index=classes).to_frame("Constant").T
            #scores of individuals
            ind_scores = X.dot(coef).add(intercept.values,axis=1)
            #concatenate
            self.coef_ = concat((intercept,coef),axis=0)

            #add score to ordered dictionary
            ind_ = OrderedDict(**ind_,**OrderedDict(scores=ind_scores))

        #remove priors log-probabilities
        if not (isinstance(self.priors, str) and self.priors == "equal"):
            class_gen, ind_gen = class_gen.sub(2*log(priors),axis=1), ind_gen.sub(2*log(priors),axis=1)

        #convert to ordered dictionary
        classes_, ind_ = OrderedDict(**classes_, **OrderedDict(mahal=class_mahal,gen=class_gen)), OrderedDict(**ind_,**OrderedDict(mahal=ind_mahal,gen=ind_gen))
        #convert to namedtuple
        self.classes_, self.ind_ = namedtuple("classes",classes_.keys())(*classes_.values()), namedtuple("ind",ind_.keys())(*ind_.values())   
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #multivariate goodness of fit - diagnostic test
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standard deviation
        stdev = DataFrame(c_[sqrt(diag(tcov)),sqrt(diag(pwcov)),sqrt(diag(bcov))],columns=["Total Std. Dev.","Pooled Std. Dev.","Between Std. Dev."],index=features)
        #compute univariate analysis of variance - ANOVA
        anova = concat((stdev,univ_test(X,y)),axis=1)
        #average R-Square
        unwavg_rsq, wavg_rsq = average(anova.loc[:,"R-Square"].values), average(anova.loc[:,"R-Square"].values,weights=diag(tcovb))
        avg_rsq = DataFrame([[unwavg_rsq,wavg_rsq]],index=["Average R-Square"],columns=["Unweighted","Weighted by Variance"])
        #compute multivariate analysis of variance - MANOVA
        manova = MANOVA.from_formula(formula="{}~{}".format("+".join(features),"+".join([target])), data=concat((X,y),axis=1)).mv_test(skip_intercept_test=True).summary_frame
        manova.index = manova.index.droplevel()
        #convert to dictionary
        statistics_ = OrderedDict(anova=anova,manova=manova,average_rsq=avg_rsq) 
    
        #add if linear discriminant analysis
        if self.method == "linear":
            #performance
            performance = diagnostics(Vb=tcovb,Wb=pwcovb,n_samples=n_samples,n_classes=n_classes)
            #update classes
            statistics_ = OrderedDict(**statistics_,**OrderedDict(performance=performance))
        
        #convert to namedtuple
        self.statistics_ = namedtuple("statistics",statistics_.keys())(*statistics_.values()) 

        self.model_ = "discrim"
        return self
    
    def fit_transform(self,X,y):
        """
        Fit to data, then transform it

        Fits transformer to ``X`` and returns a transformed version of samples.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        y : Series of shape (n_samples,)
            Target values. True labels for ``X``.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_classes)
            Transformed samples. 
        """
        #fit discriminant analysis model
        self.fit(X,y)
        #chi-square pvalue and tolerance threshold
        p_value, tol = self.cov_.test.iloc[0,6], self.call_.tol
        if self.method == "quad" and  p_value <= tol:
            raise NotImplementedError("Since the Chi-Square value is significant at the {} level.'fit_transform' method cannot be used.".format(tol))
        return self.ind_.scores
    
    def transform(self,X):
        """
        Project data to maximize class separation

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_classes)
            Transformed data, where ``n_samples`` is the number of samples and ``n_classes`` is the number of classes.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #chi-square pvalue and tolerance threshokd
        p_value, tol = self.cov_.test.iloc[0,6], self.call_.tol
        if self.method == "quad" and  p_value <= tol:
            raise NotImplementedError("Since the Chi-Square value is significant at the {} level.'transform' method cannot be used.".format(tol))

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X=X)

        #set index name as None
        X.index.name = None

        #check if X contains original features
        if not set(self.call_.Xtot.columns).issubset(X.columns):
            raise ValueError("The names of the features is not the same as the ones in the active features of the DISCRIM result")
        
        #select original features
        X = X[self.call_.Xtot.columns]

        #split X
        split_X = splitmix(X)
        #extract elements
        X_quanti, X_quali, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.k1, split_X.k2

        #initialize DataFrame
        Xcod = DataFrame(index=X.index,columns=self.call_.X.columns).astype(float)

        #check if numerics variables
        if n_quanti > 0:
            #replace with numerics columns
            Xcod.loc[:,X_quanti.columns] = X_quanti
        
        #check if categorical variables      
        if n_quali > 0:
            #active categorics
            categorics = [x for x in self.call_.X.columns if x not in self.call_.Xtot.columns]
            #replace with dummies
            Xcod.loc[:,categorics] = tab_disjunctive(X=X_quali,dummies_cols=categorics,prefix=True,sep="")
        
        #remove non selected variables
        Xcod = Xcod.loc[:,list(self.cov_.total.index)] 
        #multiply by linear discriminant analysis coefficients
        return Xcod.dot(self.coef_.iloc[1:,:]).add(self.coef_.iloc[0,:].values,axis=1)