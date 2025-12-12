# -*- coding: utf-8 -*-
from pandas import DataFrame, concat
from collections import namedtuple, OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

#interns functions
from ._base import _BaseDA
from ._mpca import MPCA
from ._discrim import DISCRIM

class MDA(_BaseDA):
    """
    Mixed Discriminant Analysis (MDA)

    Performs a linear discrimination analysis (LDA) on mixed predictors. It's a classical linear discriminant analysis carried out on the principal factors of a mixed principal component analysis (MPCA) of explanatory variables.
    Discriminat analysis on mixed predictors consists in two steps:

    1. Computation of mixed principal component analysis (MPCA) on features.
    2. Performns linear discriminant analysis (LDA) on principal components extract in step 1.

    Parameters
    ----------
    n_components : int or None, default = 2
        Number of components to keep. If ``None``, keep all the components.

    priors : str, 1-D array or Series of shape (n_classes,), default = None
        The priors statement specifies the class prior probabilities of group membership, possibles values: 

        - 'equal' to set the prior probabilities equal.
        - 'prop' to set the prior probabilities proportional to the sample sizes.
        - 1-D array or Series which specify the prior probability for each level of the classification variable.
    
    classes : None, tuple or list, default = None
        Name of level in order to return. If None, classes are sorted in unique values in y.

    Returns
    -------
    call_ : NamedTuple
        Call informations.
        
        - X : DataFrame of shape (n_samples, n_features)
            Training data
        - y : Series of shape (n_samples,)
            Target values. True values for `X`.
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
        - max_components : int
            Maximum number of components.
        - n_components : int
            Number of components kept.

    cancoef_ : NamedTuple
        Canonical coefficients:

        - standardized : DataFrame of shape (n_variables, n_components)
            The standardized canonical coefficients
        - raw : DataFrame of shape (n_variables+1, n_componets)
            The raw canonical coefficients
        - projection : DataFrame of shape (n_variables+1, n_components)
            The projection canonical coefficients.

    coef_ : NamedTuple
        Linear discriminant coefficients:

        - standardized : DataFrame of shape (n_variables, n_classes)
            The standardized coefficients.
        - raw : DataFrame of shape (n_variables+1, n_classes)
            The raw coefficients.
        - projection : DataFrame of shape (n_variables+1, n_classes)
            The projection coefficients.

    ind_ : NamedTuple
        Individuals informations:

        - scores : DataFrame of shape (n_samples, n_classes)
            The scores of individuals.
        - projection : DataFrame of shape (n_samples, n_classes)
            The projection of individuals.

    model_ : str, default = 'mda'
        The model fitted.

    pipe_ : a sequence of data transformers with two named_steps :
        - mpca : mixed principal components analysis (MPCA)
        - lda : linear discriminant analysis (LDA)

    See also
    --------
    :class:`~discrimintools.GFA`
        General Factor Analysis (GFA)
    :class:`~discrimintools.GFALDA`
        General Factor Analysis Linear Discriminant Analysis (GFALDA)
    :class:`~discrimintools.MPCA`
        Mixed Principal Component Analysis (MPCA)
    :class:`~discrimintools.summaryGFA`
        Printing summaries of General Factor Analysis model.
    :class:`~discrimintools.summaryGFALDA`
        Printing summaries of General Factor Analysis Linear Discriminant Analysis model.
    :class:`~discrimintools.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.
    :class:`~discrimintools.summaryMPCA`
        Printing summaries of Mixed Principal Component Analysis model.

    References
    ----------
    [1] Abdesselam, R. (2006), « `Mixed principal component analysis`_ ». In M. Nadif & F. X. Jollis (Eds), Actes des XIIIémes Rencontres SFC-2006 (pp. 27-31). Metz, France.
    
    [2] Abdesselam, R. (2010), « `Discriminant Analysis on Mixed Predictors`_ ». 

    .. _Mixed principal component analysis: https://perso.univ-lyon2.fr/~rabdesse/fr/Publications/IFCS-2006.pdf
    .. _Discriminant Analysis on Mixed Predictors: https://perso.univ-lyon2.fr/~rabdesse/fr/Publications/ADMixte.pdf

    See also
    --------
    

    Examples
    --------
    >>> from discrimintools.datasets import load_dataset
    >>> from discrimintools import MDA
    >>> D = load_dataset("heart") # load training data
    >>> y, X = D["disease"], D.drop(columns=["disease"]) # split into X and y
    >>> clf = MDA(n_components=5)
    >>> clf.fit(X,y)
    MDA(n_components=5)
    ```
    """
    def __init__(
            self, n_components = 2, priors = None, classes = False
    ):
        self.n_components = n_components
        self.priors = priors
        self.classes = classes

    def decision_function(self,X) -> DataFrame:
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
        return self.pipe_.decision_function(X)
    
    def fit(self,X,y) -> DataFrame:
        """
        Fit the Mixed Discriminant Analysis model

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training Data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of features.
        
        y : Series of shape (n_samples,). 
            Target values. True labels for ``X``.

        Returns
        -------
        self : object
            Fitted estimator
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #mixed discriminant analysis (MDA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.pipe_ = Pipeline([("mpca",MPCA(n_components=self.n_components)),
                               ("lda",DISCRIM(method='linear',priors=self.priors,classes=self.classes,warn_message=False))]).fit(X, y)
        
        #extract separate fitted models
        mpca, clf = self.pipe_["mpca"], self.pipe_["lda"]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical discriminant coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample standardized canonical coefficients
        std_cancoef = DataFrame(mpca.svd_.V,index=mpca.var_.coord.index,columns=mpca.var_.coord.columns)
        #total-sample raw (unstandardized) canonical coefficients
        raw_cancoef = concat((std_cancoef.T.dot(-(mpca.call_.center + mpca.call_.xc_center)/mpca.call_.xc_scale).to_frame("Constant").T, std_cancoef.div(mpca.call_.xc_scale,axis=0)),axis=0)
        #projection function coefficients
        proj_cancoef = mpca.var_.coord.div(mpca.call_.denom,axis=0).div(mpca.svd_.vs[:mpca.call_.n_components],axis=1)
        #convert to ordered dictionary
        cancoef_ = OrderedDict(standardized=std_cancoef,raw=raw_cancoef,projection=proj_cancoef)
        #convert to namedtuple
        self.cancoef_ = namedtuple("cancoef",cancoef_.keys())(*cancoef_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##classification functions
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize coefficients
        std_coef = concat((clf.coef_.iloc[0,:].to_frame().T,std_cancoef.dot(clf.coef_.iloc[1:,:])),axis=0)
        #concatenate
        raw_coef = raw_cancoef.dot(clf.coef_.iloc[1:,:])
        #add constant to canonical constant
        raw_coef.iloc[0,:] = raw_coef.iloc[0,:].to_frame().T.add(clf.coef_.iloc[0,:].to_frame().T,axis=1)
        #using projection
        proj_coef = concat((clf.coef_.iloc[0,:].to_frame().T,proj_cancoef.dot(clf.coef_.iloc[1:,:])),axis=0)
        #convert to ordered dictionary
        coef_ = OrderedDict(standardized=std_coef,raw=raw_coef,projection=proj_coef)
        #convert to namedtuple
        self.coef_ = namedtuple("coef",coef_.keys())(*coef_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ##Individuals informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #score with unstandardized coefficients
        ind_ustd_scores = mpca.call_.Xcod.dot(raw_coef.iloc[1:,:]).add(raw_coef.iloc[0,:].values,axis=1)
        #score with unstandardized coefficients
        ind_proj_scores = mpca.call_.Xcod.dot(proj_coef.iloc[1:,:]).add(proj_coef.iloc[0,:].values,axis=1)
        #convert to ordered dictionary
        ind_ = OrderedDict(scores=ind_ustd_scores,projection=ind_proj_scores)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())     

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=X,X=X,y=y,target=clf.call_.target,features=list(X.columns),classes=clf.call_.classes,priors=clf.call_.priors,n_samples=clf.call_.n_samples,n_features=clf.call_.n_features,n_classes=clf.call_.n_classes,
                            max_components = mpca.call_.max_components,n_components=mpca.call_.n_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #set model name
        self.model_ = "mda"
        return 

    def fit_transform(self,X,y) -> DataFrame:
        """
        Fit to data, then transform it

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training Data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        y : Series of shape (n_samples,)
            Target values. True labels for ``X``.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_classes)
                Transformed data, where ``n_samples`` is the number of samples and ``n_classes`` is the number of classes
        """
        self.fit(X,y)
        return self.ind_.scores
    
    def transform(self,X) -> DataFrame:
        """
        Project data to maximize class separation

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_classes)
                Transformed data, where ``n_classes`` is the number of classes.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)
        return self.pipe_.transform(X)