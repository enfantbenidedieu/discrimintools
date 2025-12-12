# -*- coding: utf-8 -*-
from numpy import average, cov, sqrt, array, ones, zeros, linalg, insert, diff, c_,cumsum,nan,diag,unique
from pandas import Series, DataFrame, concat, get_dummies
from itertools import repeat, chain
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from .functions.utils import check_is_dataframe
from .functions.preprocessing import preprocessing
from .functions.splitmix import splitmix
from .functions.concat_empty import concat_empty
from .functions.gsvd import gsvd
from .functions.tab_disjunctive import tab_disjunctive

class GFA(TransformerMixin,BaseEstimator):
    """
    General Factor Analysis (GFA)

    General factor analysis refers to dimensionality reduction technique such as PCA, CA, MCA, FAMD which helps to reduce the number of features in a 
    dataset while keeping the most important informations. For more about generalized factor analysis, see `scientisttools <https://pypi.org/project/scientisttools/>`_.

    Parameters
    ----------
    n_components : int or None, default = 2
        Number of components to keep. If None, keep all the components.

    Returns
    -------
    call_ : NamedTuple
        Call informations:
        
        - Xtot : DataFrame of shape (n_samples, n_columns)
            Input data.
        - X : DataFrame of shape (n_samples, n_features)
            Training data.
        - dummies : None or DataFrame
            Disjunctive data.
        - k1 : int.
            Number of numerics columns.
        - k2 : int.
            Number of categorical columns.
        - Z : DataFrame of shape (n_samples, n_vars)
            Standardize data.
        - Zc : DataFrame of shape (n_samples, n_vars)
            Csentered standardize data.
        - ind_weights : Series of shape (n_samples,)
            Individuals weights.
        - var_weights : Series of shape (n_vars,)
            Variables weights.
        - center : Series of shape (n_vars,)
            Mean of variables.
        - z_center : Series of shape (n_vars,)
            Mean of recoding variables. 
        - scale : Series of shape (n_vars,)
            Scale of variables.
        - denom : Series of shape (n_vars,)
            Number of variables.
        - max_components : int.
            Maximum number of components.
        - n_components : int:
            Number of components kept.

    eig_ : DataFrame of shape (max_components, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    ind_ : NamedTuple
        Individuals informations:

        - coord : DataFrame of shape (n_samples, n_components)
            The individuals coordinates.

    model_ : str, default = 'gfa'
        The model fitted.

    svd_ : NamedTuple
        Generalized singular values decomposition:
        
        * vs : 1-D array of shape (max_components,)  
            The singular values.
        * U : 2-D array of shape (n_samples, n_components)
            The left singular vectors.
        * V : 2-D array of shape (n_vars, n_components)
            The right singular vectors.

    var_ : NamedTuple
        Variables informations:
    
        - coord : DataFrame of shape (n_vars, n_components)
            The variables coordinates.

    See also
    --------
    :class:`~discrimintools.GFALDA`
        General Factor Analysis Linear Discriminant Analysis (GFALDA)
    :class:`~discrimintools.MDA`
        Mixed Discriminant Analysis (MDA)
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
    [1] Bry X. (1996), « Analyses factorielles multiple », Economica.

    [2] Bry X. (1999), « Analyses factorielles simples », Economica.

    [3] Escofier B., Pagès J. (2023), « Analyses Factorielles Simples et Multiples », 5ed, Dunod
    
    [5] Husson, F., Le, S. and Pages, J. (2010), « Exploratory Multivariate Analysis by Example Using R, Chapman and Hall.
    
    [6] Lebart Ludovic, Piron Marie, & Morineau Alain (2006), « `Statistique Exploratoire Multidimensionnelle`_ », Dunod, Paris 4ed.
    
    [7] Pagès J. (2013), « Analyse factorielle multiple avec R : Pratique R », EDP sciences
    
    [8] Rakotomalala, R. (2020), « `Pratique des Méthodes Factorielles avec Python`_ », Université Lumière Lyon 2. Version 1.0.

    [8] Saporta Gilbert (2011), « `Probabilités, Analyse des données et Statistiques`_ », Editions TECHNIP, 3ed.
    
    [9] Tenenhaus, M. (2006), « Statistique : Méthodes pour décrire, expliquer et prévoir. Dunod.

    .. _Statistique Exploratoire Multidimensionnelle: https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-12/010038111.pdf
    .. _Pratique des Méthodes Factorielles avec Python: https://hal.science/hal-04868625v1/document
    .. _Probabilités, Analyse des données et Statistiques: https://en.pdfdrive.to/dl/probabilites-analyses-des-donnees-et-statistiques

    Examples
    --------
    >>> from discrimintools.datasets import load_alcools, load_canines, load_heart
    >>> from discrimintools import GFA
    >>> #principal components analysis (PCA)
    >>> D = load_alcools("train") # load training data
    >>> X = D.drop(columns=["TYPE"]) # extract X
    >>> clf = GFA()
    >>> clf.fit(X)
    GFA()
    >>> #multiple correspondence analysis (MCA)
    >>> D = load_canines("train") # load training data
    >>> X = D.drop(columns=["Fonction"]) # extract X
    >>> clf = GFA()
    >>> clf.fit(X)
    GFA()
    >>> #factor analysis of mixed data (FAMD)
    >>> D = load_heart("subset") # load subset data
    >>> X = D.drop(columns=["disease"]) # extract X
    >>> clf = GFA()
    >>> clf.fit(X)
    GFA()
    """
    def __init__(
            self,n_components=2
    ):
        self.n_components = n_components

    def fit(self,X,y=None):
        """
        Fit the General Factor Analysis Model

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        y : None
            y is ignored.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

        #make a copy of original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X)

        #split X 
        split_X = splitmix(X=X)
        #extract all elements
        X_quanti, X_quali, n_samples, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.n, split_X.k1, split_X.k2

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals weights
        ind_weights = Series(ones(n_samples)/n_samples,index=X.index,name="weight")

        #initialize
        Xcod, center, scale, var_weights, denom, dummies = None, None, None, None, None, None
        if n_quanti > 0:
            #compute weighted average and standard deviation
            center1 = Series(average(X_quanti,axis=0,weights=ind_weights),index=X_quanti.columns,name="center")
            scale1 = Series([sqrt(cov(X_quanti.iloc[:,i],ddof=0,aweights=ind_weights)) for i in range(n_quanti)],index=X_quanti.columns,name="scale")
            #numerics variables weights
            var_weights1 = Series(ones(n_quanti),index=X_quanti.columns,name="scale")
            #denom 
            denom1 = Series(repeat(n_quanti, n_quanti),index=X_quanti.columns,name="denon")
            #concatenate
            Xcod, center, scale = concat_empty(Xcod,X_quanti,axis=1), concat_empty(center,center1,axis=0), concat_empty(scale,scale1,axis=0) 
            var_weights, denom = concat_empty(var_weights,var_weights1,axis=0), concat_empty(denom,denom1,axis=0)

        if n_quali > 0:
            #disjunctive table
            dummies = concat((get_dummies(X_quali[q],dtype=int) for q in X_quali.columns),axis=1)
            #proportion of levels
            center2 = Series(array(dummies.mul(ind_weights,axis=0).sum(axis=0)),index=dummies.columns,name="center")
            #denom 
            denom2 = Series(repeat(n_quali, dummies.shape[1]),index=dummies.columns,name="denon")
            #concatenate
            Xcod, center, denom = concat_empty(Xcod,dummies,axis=1), concat_empty(center,center2,axis=0), concat_empty(denom,denom2,axis=0)
            #if no numerics variables - MCA scaling
            if n_quanti == 0:
                #scale
                scale2 = Series(center2,index=dummies.columns,name="scale")
                #levels weights
                var_weights2 = Series([x*y for x,y in zip(center,list(chain(*[repeat(i,k) for i, k in zip(ones(n_quali)/n_quali,[X_quali[q].nunique() for q in X_quali.columns])])))],index=dummies.columns,name="weight")
            #if numerics variables - FAMD scaling
            else:
                #scale
                scale2 = Series(sqrt(center2),index=dummies.columns,name="scale")
                #levels weights
                var_weights2 = Series(ones(dummies.shape[1]),index=dummies.columns,name="weight")
            #concatenate
            scale, var_weights =  concat_empty(scale,scale2,axis=0), concat_empty(var_weights,var_weights2,axis=0)
            
        #standardization: Z = (X - center)/scale
        Z = Xcod.sub(center,axis=1).div(scale,axis=1)

        #center if both numerics and categorics features
        z_center = Series(zeros(Z.shape[1]),index=Z.columns,name="center")
        if all(x > 0 for x in [n_quanti, n_quali]):
            z_center = Series(average(Z,axis=0,weights=ind_weights),index=Z.columns,name="center")
        Zc = Z.sub(z_center,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Zc)
        max_components = int(min(linalg.matrix_rank(Q), linalg.matrix_rank(R), n_samples - 1, Zc.shape[1] - n_quali))
        #set number of components
        if self.n_components is None:
            n_components =  max_components
        elif not isinstance(self.n_components,int):
            raise ValueError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Xcod=Xcod,k1=n_quanti,k2=n_quali,Z=Z,Zc=Zc,ind_weights=ind_weights,var_weights=var_weights,
                            center=center,z_center=z_center,scale=scale,denom=denom,max_components=max_components,n_components=n_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized singular values decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized singular values decomposition (GSVD)
        self.svd_ = gsvd(X=Zc,row_weights=ind_weights,col_weights=var_weights,n_components=n_components)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eigen_values = self.svd_.vs[:max_components]**2
        difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigen_values,difference,proportion,cumsum(proportion)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = ["Can"+str(x+1) for x in range(max_components)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the individuals
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the individuals
        ind_coord = DataFrame(self.svd_.U.dot(diag(self.svd_.vs[:n_components])),index=Z.index,columns=["Can"+str(x+1) for x in range(n_components)])
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord)
        #convert to NamedTuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the variables
        var_coord = DataFrame(self.svd_.V.dot(diag(self.svd_.vs[:n_components])),index=Z.columns,columns=["Can"+str(x+1) for x in range(n_components)])
        #update levels if both numerics and categorics variables
        if all(x > 0 for x in [n_quanti, n_quali]):
            var_coord.iloc[n_quanti:,:] = var_coord.iloc[n_quanti:,:].div(sqrt(center2),axis=0).mul(self.svd_.vs[:n_components],axis=1)
        #convert to ordered dictionary
        var_ = OrderedDict(coord=var_coord)
        #convert to NamedTuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #set model name
        self.model_ = "gfa"
        return self

    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.
        
        y : None
            y is ignored.
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_components)
            Transformed values, where ``n_components`` is the number of components.
        """
        self.fit(X,y)
        return self.ind_.coord

    def transform(self,X):
        """
        Apply the dimensionality reduction on X

        X is projected on the principal components previously extracted from a training set.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_columns)
            New data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.

        Returns
        -------
        X_new : Dataframe of shape (n_samples, n_components)
            Projection of X in the principal components, where ``n_components`` is the number of components.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if X is an instance of class pd.DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_dataframe(X)

        #set index name as None
        X.index.name = None

        #check if X contains active columns
        if not set(self.call_.X.columns).issubset(X.columns):
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the GFA result")
        
        #select active columns
        X = X[self.call_.X.columns]

        #split X
        split_X = splitmix(X)
        X_quanti, X_quali, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.k1, split_X.k2

        #create code variables
        Xcod = None

        #check if numerics variables
        if X_quanti is not None :
            if self.call_.k1 != n_quanti:
                raise TypeError("The number of numerics variables must be the same")
            #concatenate
            Xcod = concat_empty(Xcod,X_quanti,axis=1)
        
        #check if categorics variables
        if X_quali is not None:
            if self.call_.k2 != n_quali:
                raise TypeError("The number of categorics variables must be the same")
            
            #test if X contains all active categorics
            new = [x for x in unique(X_quali.values) if x not in self.call_.dummies.columns]
            if len(new) > 0:
                raise ValueError("The following categories are not in the active dataset: "+",".join(new))
            
            #disjunctive table for new individuals
            dummies = tab_disjunctive(X=X_quali,dummies_cols=self.call_.dummies.columns)
            #concatenate
            Xcod = concat_empty(Xcod,dummies,axis=1)
        
        #standardization : Z = (x - center)/scale - z_center
        Z = Xcod.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).sub(self.call_.z_center,axis=1)
        #apply transition relation
        coord = Z.mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        coord.columns = ["Can"+str(x+1) for x in range(self.call_.n_components)]
        return coord