# -*- coding: utf-8 -*-
from numpy import linalg,sum, sqrt, zeros, dot, transpose,nan, log,c_,real, insert, cumsum,diff,log,array,diag, average,corrcoef
from pandas import DataFrame, Series, concat, CategoricalDtype
from pandas.api.types import is_string_dtype
from collections import namedtuple, OrderedDict
from functools import reduce
from statsmodels.multivariate.manova import MANOVA
from scipy.spatial.distance import pdist,squareform
from sklearn.utils.validation import check_is_fitted

#interns functions
from ._base import _BaseDA
from .functions.utils import check_is_dataframe, check_is_series, check_is_bool
from .functions.preprocessing import preprocessing
from .functions.model_matrix import model_matrix
from .functions.describe import describe
from .functions.sscp import sscp
from .functions.cov_to_cor_test import cov_to_cor_test
from .functions.distance import sqmahalanobis
from .functions.wcorrcoef import wcorrcoef
from .functions.univ_test import univ_test
from .functions.diagnostics import diagnostics
from .functions.lrtest import lrtest
from .functions.splitmix import splitmix
from .functions.tab_disjunctive import tab_disjunctive

class CANDISC(_BaseDA):
    """
    Canonical Discriminant Analysis (CANDISC)

    Canonical discriminant analysis is a dimension-reduction technique related to principal component analysis and canonical correlation. 
    The methodology that is used in deriving the canonical coefficients parallels that of a one-way multivariate analysis of variance (MANOVA). 
    MANOVA tests for equality of the mean vector across class levels. Canonical discriminant analysis finds linear combinations of the quantitative variables
    that provide maximal separation between classes or groups. Given a classification variable and several
    quantitative variables, the CANDISC procedure derives `canonical variables`, which are linear combinations
    of the quantitative variables that summarize between-class variation in much the same way that principal
    components summarize total variation.

    The :class:`~discrimintools.discriminant_analysis.CANDISC` procedure performs a canonical discriminant analysis, computes squared Mahalanobis
    distances between class means, and performs both univariate and multivariate one-way analyses of variance.
    
    Parameters
    ----------
    n_components : int or `None <https://docs.python.org/3/library/constants.html#None>`_, default = 2
        Number of components to keep. If `None <https://docs.python.org/3/library/constants.html#None>`_ set all components are kept::

    classes : `None <https://docs.python.org/3/library/constants.html#None>`_, tuple or list, default = `None <https://docs.python.org/3/library/constants.html#None>`_
        Name of level in order to return. If `None <https://docs.python.org/3/library/constants.html#None>`_, classes are sorted in unique values in y.
    
    warn_message : bool, default = True
        Show warning messages. Raise a warning without making the program crash.

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

    cancoef_: NamedTuple
        Canonical coefficients:

        - raw : DataFrame of shape (n_features + 1, n_components)
            Raw canonical coefficients.
        - total : DataFrame of shape (n_features, n_components)
            Total canonical coefficients.
        - pooled : DataFrame of shape (n_features, n_components)
            Pooled canonical coefficients.

    cancorr_ : DataFrame of shape (n_components, 10)
        The canonical correlations test.

    classes_ : NamedTuple
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
        - coord : DataFrame of shape (n_classes, n_components)
            Class coordinates.
        - eucl : DataFrame of shape (n_classes, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_classes, n_classes) 
            The generalized squared distance to origin.

    coef_ : DataFrame of shape (n_features + 1, n_classes)
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
        - btotal : DataFrame of shape (n_features, n_features)
            Biased total-sample covariance matrix.
        - within : dict 
            Within-class covariance matrices.
        - bwithin : dict
            Biased within-class covariance matrices.
        - pooled : DataFrame of shape (n_features, n_features)
            Pooled within-class covariance matrix.
        - bpooled : DataFrame of shape (n_features, n_features)
            Biased pooled within-class covariance matrix.
        - between : DataFrame of shape (n_features, n_features)
            Between-class covariance matrix
        - bbetween : DataFrame of shape (n_features, n_features)
            biased between-class covariance matrix.

    eig_ : DataFrame of shape (n_components, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance

    ind_ : NamedTuple
        Individuals informations:

        - coord : DataFrame of shape (n_samples, n_components)
            The coordinates of individuals.
        - mahal : DataFrame of shape (n_samples, n_classes) 
            The squared Mahalanobis distance to origin.
        - eucl : DataFrame of shape (n_samples, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_samples, n_classes) 
            The generalized squared distance to origin.
        - scores : DataFrame of shape (n_samples, n_classes) 
            The total scores of individuals.

    model_ : str, default = "candisc"
        Name of model fitted.

    sscp_ : NamedTuple 
        Sum of square cross product (SSCP) matrices:

        -  total : DataFrame of shape (n_features, n_features)
            Total-sample SSCP matrix.
        -  within : dict
            Within-class SSCP matrices
        - pooled: DataFrame of shape (n_features, n_features)
            Pooled within-class SSCP matrix.
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
            The model global performance.

    summary_ : NamedTuple
        Summary informations:

        - infos : DataFrame of shape (3, 4)
            Summary informations (total sample size, number of features, number of classes, 
            total degree of freedom, within-class degree of freedom, between-class degree of freedom).
        - total : DataFrame of shape (n_features, 8)
            Total-sample statistics, see `pandas.Describe`_.
        - within : dict
            Within-class statistics

    svd_ : Namedtuple 
        Singular value decomposition:

        -  value : 1D array of shape (n_components,)
            The eigenvalues
        -  vectors : 2D array of shape (n_features, n_components)
            The eigenvectors
    
    var_ : NamedTuple 
        Variables informations (correlation):

        - total : DataFrame of shape (n_features, n_components)
            The total-sample correlation of variables with canonical dimensions.
        - pooled : DataFrame of shape (n_features, n_components)
            The pooled-within class correlation of variables with canonical dimensions.
        * between : DataFrame of shape (n_features, n_components)
            The between-class correlation of variables with canonical dimensions.

    See also
    --------
    :class:`~discrimintools.plot.fviz_candisc`
        Visualize Canonical Discriminant Analysis.
    :class:`~discrimintools.plot.fviz_candisc_biplot`
        Visualize Canonical Discriminant Analysis (CANDISC) - Biplot of individuals and variables.
    :class:`~discrimintools.plot.fviz_candisc_ind`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of individuals.
    :class:`~discrimintools.plot.fviz_candisc_var`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of variables.
    :class:`~discrimintools.plot.fviz_dist`
        Visualize distance between barycenter.
    :class:`~discrimintools.summary.summaryCANDISC`
        Printing summaries of Canonical Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryDA`
        Printing summaries of Discriminant Analysis model.

    References
    ----------
    [1] Lebart Ludovic, Piron Marie, & Morineau Alain (2006), « `Statistique Exploratoire Multidimensionnelle`_ », Dunod, Paris 4ed.

    [2] Ricco Rakotomalala (2020), « `Pratique de l'Analyse Discriminante Linéaire`_ », Version 1.0, Université Lumière Lyon 2.
    
    [3] Saporta Gilbert (2011), « `Probabilités, Analyse de données et Statistiques`_ »,  Editions TECHNIP, 3ed.

    [4] Tenenhaus Michel (2007), « Statistique - Méthodes pour décrire, expliquer et prévoir », Dunod.

    [5] Tenenhaus Michel (1996), « Méthodes statistiques en gestion », Dunod.

    [6] Tuffery Stephane (2017), « Data Mining et Statistique décisionelle », Editions TECHNIP, 5ed.
    
    [7] Tuffery Stephane (2025), « Data Science, Statistique et Machine learning », Editions TECHNIP, 6ed.

    [8] SAS/STAT User's Guide (2013), « `The CANDISC Procedure`_ », Chapter 31.

    .. _Statistique Exploratoire Multidimensionnelle: https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-12/010038111.pdf
    .. _Pratique de l'Analyse Discriminante Linéaire: https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf
    .. _Probabilités, Analyse de données et Statistiques: https://en.pdfdrive.to/dl/probabilites-analyses-des-donnees-et-statistiques
    .. _The CANDISC Procedure: https://support.sas.com/documentation/onlinedoc/stat/131/candisc.pdf
    .. _pandas.Describe: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
    
    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC
    >>> D = load_wine() # load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> XTest = load_wine("test") # load test data
    >>> print(clf.predict(XTest))
    1958    bad
    Name: prediction, dtype: object
    """
    def __init__(
            self, n_components = 2, classes = None, warn_message = True
    ):
        self.n_components = n_components
        self.classes = classes
        self.warn_message = warn_message

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
        #raw canonical coordinates
        coord = self.transform(X=X)
        #squared eulidean distance to origin
        gsqdist = concat((coord.sub(self.classes_.coord.loc[k,:],axis=1).pow(2).sum(axis=1).to_frame(k) for k in self.call_.classes),axis=1)
        #add priors log-probabiliies to squared euclidean distance
        return -0.5*gsqdist.sub(2*log(self.call_.priors),axis=1)

    def fit(self,X,y):
        """
        Fit the Canonical Discriminant Analysis model

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
        #number of class
        n_classes = len(uq_y)
        if self.classes is not None and isinstance(self.classes, (list,tuple)):
            if len(list(set(self.classes) & set(uq_y))) != n_classes:
                raise ValueError("Insert good classes")
            classes = [str(k) for k in self.classes]
        else:
            classes = uq_y
        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=classes,ordered=True))

        #number of samples and number of features
        n_samples, n_features = X.shape
        #set target and features names
        target, features = y.name, list(X.columns)
        
        #define subset of X
        X_k = {k : X.loc[y[y==k].index,:] for k in classes}
        #count and proportion
        n_k, p_k = y.value_counts(normalize=False).loc[classes],  y.value_counts(normalize=True).loc[classes]
        #set piors
        priors = Series(array(p_k),index=classes,name="priors")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components to kept
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #maximum components
        max_components = int(min(n_classes - 1, n_features))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components, max_components)

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=Xtot,X=X,y=y,target=target,features=features,classes=classes,priors=priors,n_samples=n_samples,n_features=n_features,n_classes=n_classes,
                            max_components=max_components,n_components=n_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #sample statistics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #summary information
        summary_infos = DataFrame({"infos" : ["Total Sample Size","Variables","Classes"],
                                   "Value" : [n_samples, n_features, n_classes],
                                   "DF" : ["DF Total", "DF Within Classes", "DF Between Classes"],
                                   "DF value" : [n_samples-1, n_samples - n_classes, n_classes-1]})
        #total-sample and within-class summaries
        tsummary, wsummary = describe(X), {k : describe(X_k[k]) for k in classes}
        #convert to ordered dictionary
        summary_ = OrderedDict(infos=summary_infos,total=tsummary,within=wsummary)
        #convert to namedtuple
        self.summary_ = namedtuple("summary",summary_.keys())(*summary_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #sum of square cross product (SSCP) matrix
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample and within-class SSCP matrices
        tsscp, wsscp = sscp(X=X), {k: sscp(X_k[k]) for k in classes}
        #pooled within-class SSCCP matrix
        pwsscp = reduce(lambda i , j : i + j, wsscp.values())
        #between-class SSCP matrix
        bsscp = tsscp - pwsscp
        #convert to ordered dictionary
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
        #pooled within-class and biased pooled within-class covariance matrices
        pwcov, pwcovb = pwsscp.div(n_samples - n_classes), pwsscp.div(n_samples)
        #inverse of within-class and pooled within-class covariance matrices
        invpwcov = linalg.inv(pwcov)
        #between-class and biased between-class covariance matrices
        bcov, bcovb  = bsscp.div(n_samples*(n_classes-1)/n_classes), bsscp.div(n_samples)
        #convert to ordered dictionary
        cov_ = OrderedDict(total=tcov,btotal=tcovb,within=wcov,bwithin=wcovb,pooled=pwcov,bpooled=pwcovb,between=bcov,bbetween=bcovb)
        #convert to namedtuple
        self.cov_ = namedtuple("cov",cov_.keys())(*cov_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correlation coefficients test
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total sample and within-class correlation coefficients
        tcortest, wcortest  = cov_to_cor_test(X=tcovb,n_samples=n_samples), {k: cov_to_cor_test(X=wcovb[k],n_samples=n_k[k]) for k in classes}
        #pooled within-class and between-class correlation coefficients
        pwcortest, bcortest = cov_to_cor_test(X=pwcov,n_samples=n_samples-n_classes + 1), cov_to_cor_test(X=bcov,n_samples=n_classes)
        #convert to ordered dictionary
        cortest_ = OrderedDict(total=tcortest,within=wcortest,pooled=pwcortest,between=bcortest)
        #convert to namedtuple
        self.corr_ = namedtuple("corr",cortest_.keys())(*cortest_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classes informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #class level information
        class_infos = DataFrame(c_[n_k,p_k,priors],columns=["Frequency","Proportion","Prior Probability"],index=classes)
        class_infos["Frequency"] = class_infos["Frequency"].astype(int)
        #within-class average
        class_center = concat((X_k[k].mean(axis=0).to_frame(k) for k in classes),axis=1).T
        #total-sample standardized class means
        tcenter = class_center.sub(tsummary["mean"],axis=1).div(sqrt(diag(tcov)),axis=1).T
        #pooled within-class standardized class means
        pcenter = class_center.sub(tsummary["mean"],axis=1).div(sqrt(diag(pwcov)),axis=1).T
        #squared mahalanobis distances between class - pairwise squared distances between groups
        class_mahal = concat((sqmahalanobis(X=class_center,VI=invpwcov,mu=class_center.loc[k,:]).to_frame(k) for k in classes),axis=1)
        #convert to ordered dictionary
        classes_ = OrderedDict(infos=class_infos,center=class_center,total=tcenter,pooled=pcenter,mahal=class_mahal)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #matrix C
        C = class_center.sub(tsummary["mean"],axis=1).mul(sqrt(priors),axis=0).T
        #eigen decomposition
        eig = linalg.eig(dot(dot(C.T,linalg.inv(tcovb)),C))
        #gestion des nombres complexes
        lambd = array(sorted(real(eig.eigenvalues),reverse=True))[:n_components]
        #find index
        idx = [list(real(eig.eigenvalues)).index(x) for x in lambd]
        #reorder eigen vectors
        vector = real(eig.eigenvectors)[:,idx]
        #convert to namedtuple
        self.svd_ = namedtuple("svd",["value","vector"])(lambd,vector)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigenvalue informations
        rho = array([x/(1-x) for x in lambd])
        difference, proportion = insert(-diff(rho),len(rho)-1,nan), 100*rho/sum(rho)
        #store all informations
        self.eig_ = DataFrame(c_[rho,difference,proportion,cumsum(proportion)],columns=["Eigenvalue","Difference","Proportion","Cumulative"],index = ["Can"+str(x+1) for x in range(n_components)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical coefficients - canonical discriminant coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coefficients of features (without intercept)
        rawcoef = DataFrame(dot(dot(linalg.inv(tcovb),C),vector),index=features,columns=["Can"+str(x+1) for x in range(n_components)]).mul(sqrt((n_samples-n_classes)/(n_samples*lambd*(1-lambd))),axis=1)
        #intercept
        rawintercept = - rawcoef.T.dot(tsummary["mean"].values.reshape(-1,1))
        rawintercept.columns = ["Constant"]
        #total-sample standardized canonical coefficients
        tcan_coef = rawcoef.mul(sqrt(diag(tcov)),axis=0)
        #pooled within-class standardized canonical coefficients
        pcan_coef = rawcoef.mul(sqrt(diag(pwcov)),axis=0)
        #convert to ordered dictionary
        cancoef_ = OrderedDict(raw=concat((rawintercept.T,rawcoef),axis=0),total=tcan_coef,pooled=pcan_coef)
        #convert to namedtuple
        self.cancoef_ = namedtuple("cancoef",cancoef_.keys())(*cancoef_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correlation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correlation
        cancorr = DataFrame(c_[sqrt(lambd),lambd],columns=["Canonical Correlation","Squared Canonical Correlation"])
        #likelohood ratio test
        lr_test = DataFrame(zeros((n_components,8)),columns=["Likelihood Ratio","Approximate F value","Num DF","Den DF","Pr>F","Chi-Square","DF","Pr>Chi2"])
        for i in range(n_components):
            lr_test.iloc[-i,:] = lrtest(n_samples=n_samples,n_features=n_features,n_classes=n_classes,eigen=lambd[-(i+1):])
        lr_test = lr_test.sort_index(ascending=False).reset_index(drop=True)
        lr_test["Num DF"], lr_test["Den DF"], lr_test["DF"] = lr_test["Num DF"].astype(int), lr_test["Den DF"].astype(int), lr_test["DF"].astype(int)
        #convert to CANDISC attribute
        self.cancorr_ = concat((cancorr,lr_test),axis=1)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: squared mahalanobis distance & coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals scores
        ind_coord = X.sub(tsummary["mean"],axis=1).dot(rawcoef)
        #squared mahalanobis distance to origin
        ind_mahal = concat((sqmahalanobis(X=X,VI=invpwcov,mu=class_center.loc[k,:]).to_frame(k) for k in classes),axis=1)
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord,mahal=ind_mahal)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #others classes informations: coordinates of classes, coordinates of center of classes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates of the classes
        class_coord = concat((ind_coord.loc[y[y==k].index,:].mean(axis=0).to_frame(k) for k in classes),axis=1).T
        #squared euclidean distance between classes
        class_eucl = DataFrame(squareform(pdist(class_coord,metric="sqeuclidean")),index=classes,columns=classes)
        #squared generalized distance
        class_gen = class_eucl.sub(2*log(priors),axis=1)
        #updated classes_ dictionary
        classes_ = OrderedDict(**classes_, **OrderedDict(coord=class_coord,eucl=class_eucl,gen=class_gen))
        #convert to namedtuple
        self.classes_ = namedtuple("classes",classes_.keys())(*classes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables coordinates: total, within & between
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total correlation - total canonical Structure
        var_tcoord = DataFrame(corrcoef(x=X,y=ind_coord,rowvar=False)[:n_features,n_features:],index=features,columns=ind_coord.columns)
        #poold within correlation - Pooled Within Canonical Structure
        z1, z2 = ind_coord.sub(class_coord.loc[y.values,:].values), X.sub(class_center.loc[y.values,:].values,axis=1)
        var_pcoord = DataFrame(transpose(corrcoef(x=z1,y=z2,rowvar=False)[:n_components,n_components:]),index=features,columns=ind_coord.columns)
        #between correlation - between Canonical Structure
        var_bcoord = concat((Series([wcorrcoef(class_center[k],class_coord[l],priors) for l in ind_coord.columns],index=ind_coord.columns,name=k) for k in features),axis=1).T
        #convert to ordered dictionary
        var_ = OrderedDict(total=var_tcoord,pooled=var_pcoord,between=var_bcoord) 
        #onvert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classification function
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coefficients of classification function
        self.coef_ = self.cancoef_.raw.dot(class_coord.T)
        #update intercept
        self.coef_.iloc[0,:] = self.coef_.iloc[0,:].sub(0.5*class_coord.pow(2).sum(axis=1)).add(log(p_k))
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: scores, squared euclidean distance and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #squared euclidean distance to class center
        ind_eucl = concat((ind_coord.sub(class_coord.loc[k,:],axis=1).pow(2).sum(axis=1).to_frame(k) for k in classes),axis=1)
        #squared generalized distance
        ind_gen = ind_eucl.sub(2*log(priors),axis=1)
        #individuals scores
        ind_scores = X.dot(self.coef_.iloc[1:,:]).add(self.coef_.iloc[0,:].values,axis=1)
        #update individuals dictionary
        ind_ = OrderedDict(**ind_,**OrderedDict(eucl=ind_eucl,gend=ind_gen,scores=ind_scores))
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #others statistics
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
        #performance
        performance = diagnostics(Vb=tcovb,Wb=pwcovb,n_samples=n_samples,n_classes=n_classes)
        #convert to ordered dictionary
        statistics_ = OrderedDict(anova=anova,manova=manova,average_rsq=avg_rsq,performance=performance)   
        #convert to namedtuple
        self.statistics_ = namedtuple("statistics",statistics_.keys())(*statistics_.values())
        
        #set model name
        self.model_ = "candisc"
        return self
    
    def fit_transform(self,X,y):
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
        X_new : DataFrame of shape (n_samples, n_components)
            Transformed data, where ``n_components`` is the number of components
        """
        #fit the model
        self.fit(X,y)
        return self.ind_.coord
    
    def transform(self,X):
        """
        Apply the dimensionality reduction on X

        X is projected on the canonical components previously extracted from a training set.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            New data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_components)
            Transformed data, where ``n_components`` is the number of components.
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
            raise ValueError("The names of the features is not the same as the ones in the active features of the CANDISC result")
        
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
        
        #multiply by raw canonical coefficients
        return Xcod.dot(self.cancoef_.raw.iloc[1:,:]).add(self.cancoef_.raw.iloc[0,:],axis=1)