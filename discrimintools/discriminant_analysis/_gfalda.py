# -*- coding: utf-8 -*-
from numpy import log
from pandas import DataFrame, concat
from collections import OrderedDict, namedtuple
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist,squareform
from sklearn.utils.validation import check_is_fitted

#interns functions
from ._base import _BaseDA
from ._gfa import GFA
from ._discrim import DISCRIM

class GFALDA(_BaseDA):
    """
    General Factor Analysis Linear Discriminant Analysis (GFALDA)
    
    Performs a linear discrimination analysis on principal components. It's a classical linear discriminant analysis (LDA) carried out on the principal components of a general factor analysis (GFA) of explanatory variables.
    General factor analysis linear discriminant analysis (GFALDA) consists in two steps::

    1. Computation of general factor analysis (GFA) of explanatory variables:
        - If all features are numerics, general factor analysis (GFA) is a principal component analysis (PCA),
        - if all features are categorics, general factor analysis (GFA) is a multiple correspondence analysis (MCA),
        - if mixed features, general factor analysis (GFA) is a factor analysis of mixed data (FAMD).
    2. Computation of linear discriminant analysis (LDA) on principal components extract in step 1.

    Parameters
    ----------
    n_components : int or None, default = 2
        Number of components to keep. If ``None``, keep all the components.

    priors : str, 1-D array or Series of shape (n_classes,), default = None
        The priors statement specifies the class prior probabilities of group membership, possibles values: 

        * 'equal' to set the prior probabilities equal.
        * 'prop' to set the prior probabilities proportional to the sample sizes.
        * 1-D array or Series which specify the prior probability for each level of the classification variable.
    
    classes : None, tuple or list, default = None
        Name of level in order to return. If ``None``, classes are sorted in unique values in y.
   
    Returns
    -------
    call_ : NamedTuple
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
            Number of target values
        - max_components : int
            Maximum number of components.
        - n_components : int
            Number of components kept.

    cancoef_ : NamedTuple
        Canonical coefficients:

        - standardized : DataFrame of shape (n_variables, n_components)
            The standardized canonical coefficients.
        - raw : DataFrame of shape (n_variables+1, n_componets)
            The raw canonical coefficients.
        - projection : DataFrame of shape (n_variables+1, n_components)
            The projection canonical coefficients.

    classes_ : NamedTuple
        Classes informations:

        - coord : DataFrame of shape (n_classes, n_components)
            Class coordinates.
        - eucl : DataFrame of shape (n_classes, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_classes, n_classes) 
            The generalized squared distance to origin.

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

        - coord : DataFrame of shape (n_samples, n_components)
            Individuals coordinates.
        - scores : DataFrame of shape (n_samples, n_classes)
            The scores of individuals.
        - projection : DataFrame of shape (n_samples, n_classes)
            The projection of individuals.
        - eucl : DataFrame of shape (n_samples, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_samples, n_classes) 
            The generalized squared distance to origin.

    model_ : str, default = 'gfalda'
        The model fitted.

    pipe_ : a sequence of data transformers with two named_steps :
        - gfa : generalized factor analysis (GFA)
        - lda : linear discriminant analysis (LDA)

    See also
    --------
    :class:`~discrimintools.discriminant_analysis.GFA`
        General Factor Analysis (GFA)
    :class:`~discrimintools.discriminant_analysis.MDA`
        Mixed Discriminant Analysis (MDA)
    :class:`~discrimintools.discriminant_analysis.MPCA`
        Mixed Principal Component Analysis (MPCA)
    :class:`~discrimintools.summary.summaryGFA`
        Printing summaries of General Factor Analysis model.
    :class:`~discrimintools.summary.summaryGFA`
        Printing summaries of General Factor Analysis model.
    :class:`~discrimintools.summary.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryMPCA`
        Printing summaries of Mixed Principal Component Analysis model.

    References
    ----------
    [1] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire`_ », Université Lumière Lyon 2, Version 1.0.

    .. _Pratique de l'Analyse Discriminante Linéaire: https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf

    Examples
    --------
    >>> from discrimintools.datasets import load_alcools, load_vote, load_heart
    >>> from discrimintools import GFALDA
    >>> #PCA + LDA = PCALDA
    >>> D = load_alcools("train")
    >>> y, X = D["TYPE"], D.drop(columns=["TYPE"])
    >>> clf = GFALDA()
    >>> clf.fit(X,y) 
    GFALDA()
    >>> #MCA + LDA = DISQUAL
    >>> D = load_vote("train")
    >>> y, X = D["group"], D.drop(columns=["group"])
    >>> clf = GFALDA()
    >>> clf.fit(X,y)
    GFALDA()
    >>> #FAMD + LDA = DISMIX
    >>> D = load_heart("subset")
    >>> y, X = D["disease"], D.drop(columns=["disease"])
    >>> clf = GFALDA()
    >>> clf.fit(X,y)
    GFALDA()
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

    def fit(self,X,y):
        """
        Fit the General Factor Analysis Linear Discriminant Analysis Model

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training Data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        y : Series of shape (n_samples,)
            Target values. True labels for ``X``.

        Returns
        -------
        self : object
            Fitted estimator
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #discriminant analysis on principal components (DAPC)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.pipe_ = Pipeline([("gfa",GFA(n_components=self.n_components)),
                               ("lda",DISCRIM(method="linear",priors=self.priors,classes=self.classes,warn_message=False))]).fit(X, y)
        
        #extract separate fitted models
        gfa, clf = self.pipe_["gfa"], self.pipe_["lda"]

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=X,X=X,y=y,target=clf.call_.target,features=list(X.columns),classes=clf.call_.classes,priors=clf.call_.priors,n_samples=clf.call_.n_samples,n_features=clf.call_.n_features,n_classes=clf.call_.n_classes,
                            max_components = gfa.call_.max_components,n_components=gfa.call_.n_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical discriminant coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample standardized canonical coefficients
        std_cancoef = DataFrame(gfa.svd_.V,index=gfa.var_.coord.index,columns=gfa.var_.coord.columns)
        #total-sample raw (unstandardized) canonical coefficients
        raw_cancoef = concat((std_cancoef.T.dot(-(gfa.call_.center/gfa.call_.scale + gfa.call_.z_center)).to_frame("Constant").T, std_cancoef.div(gfa.call_.scale,axis=0)),axis=0)
        #projection function coefficients
        proj_cancoef = gfa.var_.coord.div(gfa.call_.denom,axis=0).div(gfa.svd_.vs[:gfa.call_.n_components],axis=1)
        #convert to ordered dictionary
        cancoef_ = OrderedDict(standardized=std_cancoef,raw=raw_cancoef,projection=proj_cancoef)
        #convert to namedtuple
        self.cancoef_ = namedtuple("cancoef",cancoef_.keys())(*cancoef_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification functions
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
        #individuals coordinates
        ind_coord = gfa.ind_.coord
        #score with unstandardized coefficients
        ind_scores = gfa.call_.Xcod.dot(raw_coef.iloc[1:,:]).add(raw_coef.iloc[0,:].values,axis=1)
        #score with unstandardized coefficients
        ind_proj = gfa.call_.Xcod.dot(proj_coef.iloc[1:,:]).add(proj_coef.iloc[0,:].values,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classes informations: coordinates, squared euclidean distance and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classes coordinates
        class_coord = concat((ind_coord.loc[y[y==k].index,:].mean(axis=0).to_frame(k) for k in clf.call_.classes),axis=1).T
        #squared euclidean distance between classes
        class_eucl = DataFrame(squareform(pdist(class_coord,metric="sqeuclidean")),index=clf.call_.classes,columns=clf.call_.classes)
        #squared generalized distance
        class_gen = class_eucl.sub(2*log(clf.call_.priors),axis=1)
        #convert to ordered dictionary
        classes_ = OrderedDict(infos=clf.classes_.infos,coord=class_coord,eucl=class_eucl,gen=class_gen)
        #convert to namedtuple
        self.classes_ = namedtuple("classes",classes_.keys())(*classes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals additional informations: squared euclidean distance between classes barycenters and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #squared euclidean distance to class center
        ind_eucl = concat((ind_coord.sub(class_coord.loc[k,:],axis=1).pow(2).sum(axis=1).to_frame(k) for k in clf.call_.classes),axis=1)
        #squared generalized distance
        ind_gen = ind_eucl.sub(2*log(clf.call_.priors),axis=1)
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord,scores=ind_scores,projection=ind_proj,eucl=ind_eucl,gen=ind_gen)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        self.model_ = "gfalda"
        return self
    
    def fit_transform(self,X,y):
        """
        Fit to data, then transform it

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training Data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of features.

        y : Series of shape (n_samples,)
            Target values. True labels for ``X``.

        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_classes)
                Transformed data, where ``n_classes`` is the number of classes.
        """
        self.fit(X,y)
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
                Transformed data, where ``n_classes`` is the number of classes.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)
        return self.pipe_.transform(X)