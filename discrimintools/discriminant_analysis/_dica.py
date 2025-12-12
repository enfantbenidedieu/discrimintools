# -*- coding: utf-8 -*-
from numpy import array, linalg, c_, log, sqrt, unique, insert, diff, nan, cumsum, diag
from pandas import Series, CategoricalDtype, concat, DataFrame
from pandas.api.types import is_string_dtype
from collections import OrderedDict, namedtuple
from scipy.spatial.distance import pdist,squareform
from sklearn.utils.validation import check_is_fitted

#interns functions
from ._base import _BaseDA
from .functions.utils import check_is_dataframe, check_is_series
from .functions.preprocessing import preprocessing
from .functions.gsvd import gsvd
from .functions.univ_test import univ_test
from .functions.tab_disjunctive import tab_disjunctive

class DiCA(_BaseDA):
    """
    Discriminant Correspondence Analysis (DiCA)
    
    Performs Discriminant Correspondence Analysis to classify each observation into groups
    
    This component performs a canonical discriminant analysis (CANDISC) where we want to characterize the groups of individuals (described by a discrete target attribute) from a set of discrete descriptors.
    This approach is based on a correspondence analysis (CA) on an overall crosstab which os a concatenation of individual crosstabs between the target attribute with each predictive attribute (see [4]_).
    We obtain factor scores both for values of the target attribute and the input ones which enable to explain the relationship between the variables. We obtain also a factor score coefficients which enable to calculate the coordinates 
    of new individuals from their indicator vector description.
    
    Parameters
    ----------
    n_components : int, default = 2
        Number of components to keep. If None, keep all the components.
    
    classes : None, tuple or list, default = None
        Name of level in order to return. If None, classes are sorted in unique values in y.

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
            Priors probabilities
        - n_samples : int
            Number of samples.
        - n_features : int
            Number of features.
        - n_classes : int
            Number of target values.
        - N : DataFrame of shape (n_classes,  n_categories)
            The contingence table for correspondence analysis.
        - Z : DataFrame of shape (n_classes, n_categories)
            The standardized data.
        - total : int
            The total size : ``total_size = n_samples * n_features``
        - row_marge : Series of shape (n_classes,)
            The row margins of frequencies table.
        - col_marge : Series of shape (n_categories,)
            The column margins of frequencies table.
        - max_components : int
            Maximum number of components.
        - n_components : int
            Number of components kept.

    cancoef_ : NamedTuple 
        Coefficients of discriminant correspondence analysis:

        - standardize : DataFrame of shape (n_categories, n_components)
            The standardized coefficients.
        - projection : DataFrame of shape (n_categories, n_components)
            The projection coefficients.

    cancorr_ : DataFrame of shape (2, 4)
        The canonical correlations test.

    classes_ : NamedTuple 
        Classes informations:

        - infos : DataFrame of shape (n_classes, 3)
            class level information (frequency, proportion, prior probability).
        - coord : DataFrame of shape (n_classes, n_components)
            The class coordinates.
        - eucl : DataFrame of shape (n_classes, n_classes)
            The squared Euclidean distances between classes.
        - gen : DataFrame of shape (n_classes, n_classes)
            The squared generalized distances between classes.

    eig_ : DataFrame of shape (n_components, 4)
        The eigenvalues, the difference between each eigenvalue, the percentage of variance and the cumulative percentage of variance    
    
    ind_ : NamedTuple
        Individuals informations:

        - coord : DataFrame of shape (n_samples, n_components)
            The coordinates of individuals.
        - eucl : DataFrame of shape (n_samples, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame of shape (n_samples, n_classes)
            The generalized squared distance to origin.

    model_ : str, default = 'dica'
        The model fitted.

    svd_ : Namedtuple 
        Generalized singular value decomposition (GSVD):

        - svd : 1D array of shape (n_components,)
            The singular values.
        - U : 2D array of shape (n_samples, n_components)
            The left singular vectors of generalized singular values decomposition.
        - V : 2D array of shape (n_categories, n_components)
            The right singular vectors of generalized singular values decomposition.
    
    var_ : NamedTuple 
        Variables informations:

        - coord : DataFrame of shape (n_categories, n_components)
            The coordinates of variables.
        - eta2 : DataFrame of shape (n_features, n_components)
            The square correlation ratio - eta2.

    See also
    --------
    :class:`~discrimintools.fviz_dica`
        Visualize Discriminant Correspondence Analysis Analysis.
    :class:`~discrimintools.fviz_dica_biplot`
        Visualize Discriminant Correspondence Analysis (DiCA) - Biplot of individuals and variables.
    :class:`~discrimintools.fviz_dica_ind`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of individuals.
    :class:`~discrimintools.fviz_dica_quali_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of qualitative variables.
    :class:`~discrimintools.fviz_dica_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of variables/categories.
    :class:`~discrimintools.fviz_dist`
        Visualize distance between barycenter.
    :class:`~discrimintools.summaryDiCA`
        Printing summaries of Discriminant Correspondence Analysis model.
    :class:`~discrimintools.summaryDA`
        Printing summaries of Discriminant Analysis model.

    References
    ----------
    [1] Abdi, H., and Williams, L.J. (2010). `Principal component analysis <https://pzs.dstu.dp.ua/DataMining/pca/bibl/PCA.pdf>`_. Wiley Interdisciplinary Reviews: Computational Statistics, 2, 433-459.
    
    [2] Abdi, H. and Williams, L.J. (2010). `Correspondence analysis <https://personal.utdallas.edu/~herve/abdi-AB2018_CA.pdf>`_. In N.J. Salkind, D.M., Dougherty, & B. Frey (Eds.): Encyclopedia of Research Design. Thousand Oaks (CA): Sage. pp. 267-278.
    
    [3] Abdi, H. (2007). `Singular Value Decomposition (SVD) and Generalized Singular Value Decomposition (GSVD) <https://www.cimat.mx/~alram/met_num/clases/Abdi-SVD2007-pretty.pdf>`_. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics.Thousand Oaks (CA): Sage. pp. 907-912.
    
    .. [4] Abdi, H. (2007). `Discriminant correspondence analysis <https://personal.utdallas.edu/~herve/Abdi-DCA2007-pretty.pdf>`_. In N.J. Salkind (Ed.): Encyclopedia of Measurement and Statistics. Thousand Oaks (CA): Sage. pp. 270-275
    
    [5] Ricco Rakotomalala (2020), `Pratique de l'Analyse Discriminante Linéaire <https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf>`_, Version 1.0, Université Lumière Lyon 2.

    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA
    >>> D = load_divay() # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    """
    def __init__(
            self, n_components = 2, classes = None
    ):
        self.n_components = n_components
        self.classes = classes

    def decision_function(self,X) -> DataFrame:
        """
        Apply decision function to a an input data

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            
        Returns
        -------
        C : DataFrame of shape (n_samples, n_classes)
            Decision function values related to each class, per sample.
        """
        #canonical coordinates
        coord = self.transform(X)
        #squared euclidean distance to origin
        gsqdist = concat((coord.sub(self.classes_.coord.loc[k,:],axis=1).pow(2).sum(axis=1).to_frame(k) for k in self.call_.classes),axis=1)
        #add priors log-probabilities to squared euclidean distance if priors is not equal
        return -0.5*gsqdist.sub(2*log(self.call_.priors),axis=1)
    
    def fit(self,X,y):
        """
        Fit the Discriminant Correspondence Analysis model

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

        #check if all features are categorics
        if not all(is_string_dtype(X[k]) for k in X.columns):
            raise TypeError("All features must be categorics")
        
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
        #check if all columns in X are categorics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not all (is_string_dtype(X[k]) for k in X.columns):
            raise TypeError("X mus contain categorical columns")

        #make a copy of original data
        Xtot = X.copy(deep=True)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #preprocessing
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = preprocessing(X=X)

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

        #number of samples and number of features
        n_samples, n_features = X.shape
        #set target and features names
        target, features = y.name, list(X.columns)

        #count and proportion
        n_k, p_k = y.value_counts(normalize=False).loc[classes],  y.value_counts(normalize=True).loc[classes]
        #convert to pandas Series
        priors = Series(array(y.value_counts(normalize=True).loc[classes]),index=classes,name="priors")

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #correspondence analysis (CA)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #disjunctive table
        dummies_y, dummies_x = tab_disjunctive(y).loc[:,classes], tab_disjunctive(X)
        #matrix N for correspondence analysis
        N = dummies_y.T.dot(dummies_x)
        #total
        total = n_samples*n_features
        #frequencies table
        freq = N.div(total)
        #columns and rows margins calcul
        col_marge, row_marge = freq.sum(axis=0), freq.sum(axis=1)
        col_marge.name, row_marge.name = "Margin", "Margin"

        #standardization: z_ij = (fij/(fi.*f.j)) - 1
        Z = freq.div(row_marge,axis=0).div(col_marge,axis=1).sub(1)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #set number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition (to set maximum number of components)
        Q, R = linalg.qr(Z)
        max_components = int(min(linalg.matrix_rank(Q),linalg.matrix_rank(R), n_classes - 1, dummies_x.shape[1] - 1))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise TypeError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components,max_components)

        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=Xtot,X=X,y=y,target=target,features=features,classes=classes,priors=priors,n_samples=n_samples,n_features=n_features,n_classes=n_classes,
                            N=N,Z=Z,total=total,row_marge=row_marge,col_marge=col_marge,max_components=max_components,n_components=n_components)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized singular values decomposition
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #fit generalized singular values decomposition (GSVD)
        self.svd_ = gsvd(X=Z,row_weights=row_marge,col_weights=col_marge,n_components=n_components)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #eigen values informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        eigen_values = self.svd_.vs[:max_components]**2
        difference, proportion = insert(-diff(eigen_values),len(eigen_values)-1,nan), 100*eigen_values/sum(eigen_values)
        #convert to DataFrame
        self.eig_ = DataFrame(c_[eigen_values,difference,proportion,cumsum(proportion)],columns=["Eigenvalue","Difference","Proportion (%)","Cumulative (%)"],index = ["Can"+str(x+1) for x in range(max_components)])

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classes informations: coordinates, squared euclidean distance and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #class level information
        class_infos = DataFrame(c_[n_k,p_k,priors],columns=["Frequency","Proportion","Prior Probability"],index=classes)
        class_infos["Frequency"] = class_infos["Frequency"].astype(int)
        #classes coordinates
        class_coord = DataFrame(self.svd_.U.dot(diag(self.svd_.vs[:n_components])),index=Z.index,columns=["Can"+str(x+1) for x in range(n_components)])
        #squared euclidean distance between classes
        class_eucl = DataFrame(squareform(pdist(class_coord,metric="sqeuclidean")),index=classes,columns=classes)
        #squared generalized distance
        class_gen = class_eucl.sub(2*log(priors),axis=1)
        #convert to ordered dictionary
        classes_ = OrderedDict(infos=class_infos,coord=class_coord,eucl=class_eucl,gen=class_gen)
        #convert to namedtuple
        self.classes_ = namedtuple("classes",classes_.keys())(*classes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables informations: coordinates
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coordinates for the variables
        var_coord = DataFrame(self.svd_.V.dot(diag(self.svd_.vs[:n_components])),index=Z.columns,columns=["Can"+str(x+1) for x in range(n_components)])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical discriminant functions
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample standardized canonical coefficients
        std_cancoef = DataFrame(self.svd_.V,index=Z.columns,columns=self.eig_.index[:n_components])
        #projection function coefficients
        proj_cancoef = var_coord.div(n_features*self.svd_.vs[:n_components],axis=1)
        #convert to ordered dictionary
        cancoef_ = OrderedDict(standardized=std_cancoef,projection=proj_cancoef)
        #convert to namedtuple
        self.cancoef_ = namedtuple("cancoef",cancoef_.keys())(*cancoef_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations: individuals coordinates, squared euclidean distance and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = dummies_x.dot(proj_cancoef)
        #squared euclidean distance to class center
        ind_eucl = concat((ind_coord.sub(classes_["coord"].loc[k,:],axis=1).pow(2).sum(axis=1).to_frame(k) for k in classes),axis=1)
        #squared generalized distance
        ind_gen = ind_eucl.sub(2*log(priors),axis=1)
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord,eucl=ind_eucl,gen=ind_gen)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #update variables informations: correlation ratio
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables square correlation ratio
        var_sqeta = concat(((univ_test(ind_coord,X[q])["R-Square"]).to_frame(q) for q in list(X.columns)),axis=1).T
        #convert to ordered dictionary
        var_ = OrderedDict(coord=var_coord,eta2=var_sqeta)
        #convert to namedtuple
        self.var_ = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical correlation informations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total sum of squared
        tss = ind_coord.pow(2).sum(axis=0)
        #eta squared
        sqeta = self.eig_.iloc[:n_components,0].mul(n_samples).div(tss)
        #canonical correlation
        self.cancorr_ = DataFrame(c_[self.eig_.iloc[:n_components,0],tss,sqeta,sqrt(sqeta)],columns=["Eigenvalue","Total SS", "Eta Sq.","Canonical Correlation"],index=self.eig_.index[:n_components])
        
        self.model_ = "dica"
        return self
        
    def fit_transform(self,X,y):
        """
        Fit to data, then transform it

        Fits transformer to ``X`` and returns a transformed version of ``X``.

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
            raise ValueError("The names of the features is not the same as the ones in the active features of the DiCA result")
        
        #select original features
        X = X[self.call_.Xtot.columns]

        #check if all variables are discrete
        if not all(is_string_dtype(X[k]) for k in X.columns):
            raise TypeError("All features must be categorics")
        
        #test if X contains all active categorics
        new = [x for x in unique(X.values) if x not in self.call_.N.columns]
        if len(new) > 0:
            raise ValueError("The following categories are not in the active dataset: "+",".join(new))
        
        #disjunctive table for new individuals
        dummies = tab_disjunctive(X=X,dummies_cols=self.call_.N.columns,prefix=False)   
        #multiply by canonical coefficients
        return  dummies.dot(self.cancoef_.projection)