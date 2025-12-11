
from numpy import average, cov, sqrt, array, ones, linalg, insert, diff, c_,cumsum,nan,diag,dot, unique
from pandas import Series, DataFrame, concat
from itertools import repeat, chain
from collections import OrderedDict, namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

#intern functions
from .functions.utils import check_is_dataframe
from .functions.preprocessing import preprocessing
from .functions.splitmix import splitmix
from .functions.recodecat import recodecat
from .functions.gsvd import gsvd
from .functions.tab_disjunctive import tab_disjunctive

class MPCA(TransformerMixin,BaseEstimator):
    """
    Mixed Principal Components Analysis (MPCA)
    
    Mixed Principal Components Analysis (MPCA) is a standardized principal component analysis of both quantitative variables and the transformation
    of the dummy variables associated to qualitative variables on quantitative variables through orthogonal projections of configurations of statistical units in the
    individual-space with a relational inner product. For more, see [1]_.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If ``None``, keep all the components.

    Returns
    -------
    call_ : NamedTuple
        Call informations:

        - Xtot : DataFrame of shape (n_samples, n_colums)
            Input data.
        - X : DataFrame of shape (n_samples, columns)
            Processing data.
        - dummies : DataFrame of shape (n_samples, n_categories)
            Disjunctive data.
        - Xcod : DataFrame of shape (n_samples, n_vars)
            Training data.
        - Xc : DataFrame of shape (n_samples, n_vars)
            Centered recode data.
        - Z : DataFrame of shape (n_samples, n_vars)
            Standardize recode data.
        - center : Series of shape (n_vars,)
            Average of recode data.
        - xc_center : Series of shape (n_vars,)
            Average of centered recode data.
        - xc_scale : Series of shape (n_vars,)
            Standard deviation of the centered recode data.
        - k1 : int
            Number of numerics columns.
        - k2 : int
            Number of categorical.
        - ind_weights : Series of shape (n_samples,)
            Individuals weights.
        - var_weights : Series of shape (n_vars,)
            Columns weights.
        - denom : Series of shape (n_vars,)
            number of variables.
        - max_components : int
            Maximum number of components.
        - n_components : int
            Number of components kept.

    eig_ : DataFrame of shape (max_components, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance.

    ind_ : NamedTuple
        Individuals informations:

        - coord : DataFrame of shape (n_samples, n_components)
            The individuals coordinates.

    model_ : str, defaut = 'mpca'
        The model fitted.

    svd_ : NamedTuple
        Generalized singular values decomposition:

        - vs : 1-D array of shape (max_components,)  
            The singular values.
        - U : 2-D array of shape (n_samples, n_components)
            The left singular vectors.
        - V : 2-D array of shape (n_vars, n_components)
            The right singular vectors.
    
    var_ : NamedTuple
        Variables informations:
    
        - coord : DataFrame of shape (n_vars, n_components)
            The variables coordinates.

    See also
    --------
    :class:`~discrimintools.discriminant_analysis.GFA`
        General Factor Analysis (GFA)
    :class:`~discrimintools.discriminant_analysis.GFALDA`
        General Factor Analysis Linear Discriminant Analysis (GFALDA)
    :class:`~discrimintools.discriminant_analysis.MDA
        Mixed Discriminant Analysis (MDA)
    :class:`~discrimintools.summary.summaryGFA`
        Printing summaries of General Factor Analysis model.
    :class:`~discrimintools.summary.summaryGFALDA`
        Printing summaries of General Factor Analysis Linear Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryMDA`
        Printing summaries of Mixed Discriminant Analysis model.
    :class:`~discrimintools.summary.summaryMPCA`
        Printing summaries of Mixed Principal Component Analysis model.

    References
    ----------
    .. [1] Abdesselam R. (2006), « `Analyse en Composantes Principales Mixtes`_ », CREM UMR CNRS 6211
    
    [2] Bry X. (1996), « Analyses factorielles multiple », Economica
    
    [3] Bry X. (1999), « Analyses factorielles simples », Economica
    
    [4] Escofier B., Pagès J. (2023), « Analyses Factorielles Simples et Multiples », 5ed, Dunod
    
    [5] Saporta Gilbert (2011), « `Probabilités, Analyse des données et Statistiques`_ », Editions TECHNIP, 3ed.
    
    [6] Husson, F., Le, S. and Pages, J. (2010), « Exploratory Multivariate Analysis by Example Using R », Chapman and Hall.
    
    [7] Lebart Ludovic, Piron Marie, & Morineau Alain (2006), « `Statistique Exploratoire Multidimensionnelle`_ », Dunod, Paris 4ed.
    
    [8] Pagès J. (2013), « Analyse factorielle multiple avec R : Pratique R », EDP sciences
    
    [9] Rakotomalala, R. (2020), « `Pratique des Méthodes Factorielles avec Python`_ », Université Lumière Lyon 2. Version 1.0.
    
    [10] Tenenhaus, M. (2006), « Statistique : Méthodes pour décrire, expliquer et prévoir », Dunod.

    .. _Analyse en Composantes Principales Mixtes: https://www.researchgate.net/profile/Rafik-Abdesselam/publication/5087866_Analyse_en_composantes_principales_mixte/links/0c960525d0d312c1b2000000/Analyse-en-composantes-principales-mixte.pdf
    .. _Statistique Exploratoire Multidimensionnelle: https://horizon.documentation.ird.fr/exl-doc/pleins_textes/2023-12/010038111.pdf
    .. _Pratique des Méthodes Factorielles avec Python: https://hal.science/hal-04868625v1/document
    .. _Probabilités, Analyse des données et Statistiques: https://en.pdfdrive.to/dl/probabilites-analyses-des-donnees-et-statistiques
        
    Examples
    --------
    >>> from discrimintools.datasets import load_heart
    >>> from discrimintools import MPCA
    >>> D = load_heart("subset")
    >>> X = D.drop(columns=["disease"])
    >>> clf = MPCA()
    >>> clf.fit(X)
    MPCA()
    """
    def __init__(
            self,n_components=2
    ):
        self.n_components = n_components

    def fit(self,X,y=None):
        """
        Fit the Mixed Principal Component Analysis Model

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

        #check if mixed data
        if any(x == 0 for x in [n_quanti, n_quali]):
            raise TypeError("MPCA require mixed data.")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set individuals weights
        ind_weights = Series(ones(n_samples)/n_samples,index=X.index,name="weight")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center numerics variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average mean and standard deviation
        center1 = Series(average(X_quanti,axis=0,weights=ind_weights),index=X_quanti.columns,name="center")
        #center quantitatives variables
        X1c = X_quanti.sub(center1,axis=1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #treatment of categorics variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #recode categorical variables
        rec = recodecat(X=X_quali)
        X_quali, dummies = rec.X, rec.dummies

        #covariance matrix between X and between X and Y
        Vx, Vylx = cov(X_quanti,rowvar=False,ddof=0,aweights=ind_weights), dummies.T.dot(diag(ind_weights)).dot(X1c)
        
        #compute the mean
        center2 = Series(dot(dot(dot(dot(Vylx,linalg.pinv(Vx,hermitian=True)),X1c.T),diag(ind_weights)),ones(n_samples)).T[0],index=dummies.columns,name="center")
        #center the disjunctive table
        X2c = dummies.sub(center2,axis=1)

        #concatenate
        Xcod, Xc, center = concat((X_quanti,dummies),axis=1), concat((X1c,X2c),axis=1), concat((center1,center2),axis=0)
        #variables weights
        var_weights = Series(ones(Xc.shape[1]),index=Xc.columns,name="weight")
        #denom
        denom = Series(list(chain(*[repeat(i,k) for i, k in zip([n_quanti, n_quali],[n_quanti, dummies.shape[1]])])),index=Xc.columns,name="denom")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #standardize Z = (xc - mu)/sigma
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #compute weighted average and standard deviation
        xc_center = Series(average(Xc,axis=0,weights=ind_weights),index=Xc.columns,name="weight")
        xc_scale = Series(array([sqrt(cov(Xc.iloc[:,i],ddof=0,aweights=ind_weights)) for i in range(Xc.shape[1])]),index=Xc.columns,name="weight")
        #standardization: Z = (xc - mu)/sigma
        Z = Xc.sub(xc_center,axis=1).div(xc_scale,axis=1)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_samples - 1, Z.shape[1] - n_quali))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #store call informations
        call_ = OrderedDict(Xtot=Xtot,X=X,dummies=dummies,Xcod=Xcod,Xc=Xc,Z=Z,center=center,xc_center=xc_center,xc_scale=xc_scale,k1=n_quanti,k2=n_quali,ind_weights=ind_weights,var_weights=var_weights,
                            denom=denom,max_components=max_components,n_components=n_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized singular values decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized singular values decomposition (GSVD)
        self.svd_ = gsvd(X=Z,row_weights=ind_weights,col_weights=var_weights,n_components=n_components)

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
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the variables
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the variables
        var_coord = DataFrame(self.svd_.V.dot(diag(self.svd_.vs[:n_components])),index=Z.columns,columns=["Can"+str(x+1) for x in range(n_components)])
        #convert to ordered dictionary
        var_ = OrderedDict(coord=var_coord)
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #set model name
        self.model_ = "mpca"
        return self
    
    def fit_transform(self,X,y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` is the number of samples and ``n_columns`` is the number of columns.
        
        y : None.
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
            Projection of X in the principal components where ``n_components`` is the number of the components.
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
            raise ValueError("The names of the columns is not the same as the ones in the active columns of the MPCA result")
        
        #select active columns
        X = X[self.call_.X.columns]

        #split X
        split_X = splitmix(X)
        X_quanti, X_quali, n_quanti, n_quali = split_X.quanti, split_X.quali, split_X.k1, split_X.k2
        
        #check if mixed data
        if any(x == 0 for x in [n_quanti, n_quali]):
            raise TypeError("MPCA require mixed data.")
        
        #check if len are equal
        if self.call_.k1 != n_quanti:
            raise TypeError("The number of numerics variables must be the same")
        
        #check if len are equal
        if self.call_.k2 != n_quali:
            raise TypeError("The number of categorics variables must be the same")
        
        #test if X contains all active categorics
        new = [x for x in unique(X_quali.values) if x not in self.call_.dummies.columns]
        if len(new) > 0:
            raise ValueError("The following categories are not in the active dataset: "+",".join(new))

        #disjunctive table for new individuals
        dummies = tab_disjunctive(X=X_quali,dummies_cols=self.call_.dummies.columns)
        #concatenate
        Xcod = concat((X_quanti,dummies),axis=1)
        
        #standardization: Z = (Xcod - mu1 - mu2)/sigma
        Z =  Xcod.sub(self.call_.center,axis=1).sub(self.call_.xc_center,axis=1).div(self.call_.xc_scale,axis=1)
        #apply transition relation
        coord = Z.mul(self.call_.var_weights,axis=1).dot(self.svd_.V)
        coord.columns = ["Can"+str(x+1) for x in range(self.call_.n_components)]
        return coord