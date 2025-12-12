# -*- coding: utf-8 -*-
from numpy import dot, array, linalg, where, c_, cumsum, log
from pandas import DataFrame, CategoricalDtype, get_dummies, Series, concat
from pandas.api.types import is_string_dtype
from collections import OrderedDict, namedtuple
from sklearn.cross_decomposition import PLSRegression
from scipy.spatial.distance import pdist,squareform
from sklearn.utils.validation import check_is_fitted

#interns functions
from ._base import _BaseDA
from .functions.utils import check_is_dataframe, check_is_series, check_is_bool
from .functions.preprocessing import preprocessing
from .functions.model_matrix import model_matrix
from .functions.plsrvip import plsrvip
from .functions.splitmix import splitmix
from .functions.tab_disjunctive import tab_disjunctive

class CPLS(_BaseDA):
    """
    Partial Least Squares for Classification (CPLS)

    The model filts a partial least squares for binary classification. Read more in the :ref:``.

    Parameters
    ----------
    n_components : int or None, default = 2
        Number of components to keep. Should be in ``[1, n_features]``.

    scale : bool, default = True
        Whether to scale ``X`` and ``y``.

    classes : None, tuple or list, default = None
        Name of level in order to return. If None, classes are sorted in unique values in y.

    max_iter : int, default = 500
        The maximum number of iterations for NIPALS method.

    tol : float, default = 1e-06
        The tolerance used as convergence criteria in the NIPALS method.

    var_select : bool, default = True
        Whether to applied feature selection based on variables importance in Projection for Partial Least-Squares Regression?

    threshold : float, default = 1.0
        You can use VIP to select predictor variables when multicollinearity exists among variables. 
        Variables with a VIP score greater than 'threshold' are considered important for the projection of the PLS regression.

    warn_message : bool, default = True
        Whether to show warning messages.

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
        - center : Series of shape (n_features,)
            The average of ``X``.
        - scale : Series of shape (n_features,)
            The standard deviation of ``X``.
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
        - max_iter : int
            Maximum number of iterations.
        - tol : float
            The tolerance used as convergence criteria.
        - threshold : float,
            The tolerance for variable importance in projection.

    classes_ : NamedTuple
        Classes informations:

        - infos : DataFrame of shape (n_classes, 3)
            class level information (frequency, proportion, prior probability).
        - coord : DataFrame of shape (n_classes, n_components)
            Class coordinates.
        - eucl : DataFrame of shape (n_classes, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_classes, n_classes) 
            The generalized squared distance to origin.

    coef_ : DataFrame of shape (n_features + 1, 1)
        The coefficients of the partial least squares for classification model.

    explained_variance_ : DataFrame of shape (n_components, 2)
        The explained variance and the cumulative explained variance.

    ind_ : NamedTuple
        Individuals informations:

        - coord : DataFrame of shape (n_samples, n_components)
            The transformed training simples.
        - scores : DataFrame of shape (n_samples,) 
            The total scores of individuals.
        - eucl : DataFrame of shape (n_samples, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_samples, n_classes) 
            The generalized squared distance to origin.

    model_ : str, default = 'cpls'
        The model fitted name.

    var_ : NamedTuple
        Variables informations:

        - weights : DataFrame of shape (n_features, n_components)
            The left singular vectors of the cross-covariance matrices of each iteration.
        - loadings : DataFrame of shape (n_features, n_components)
            The loadings of ``X``.
        - rotations : DataFrame of shape (n_features, n_components)
            The projection matrix used to transform ``X``.

    vip_ : NamedTuple
        Variable importance in projection information:

        - vip : Series of shape (n_features,)
            The variable importance in projection for partial least squares regression
        - selected : list
            Selected features

    See also
    --------
    :class:`~discrimintools.fviz_plsr`
        Visualize Partial Least Squares Regression.
    :class:`~discrimintools.fviz_plsr_ind`
        Visualize Partial Least Squares Regression - Graph of individuals.
    :class:`~discrimintools.fviz_plsr_var`
        Visualize Partial Least Squares Regression - Graph of variables.
    :class:`~discrimintools.fviz_dist`
        Visualize distance between barycenter.
    :class:`~discrimintools.summaryCPLS`
        Printing summaries of Partial Least Squares for Classification model.
    :class:`~discrimintools.summaryDA`
        Printing summaries of Discriminant Analysis model.

    References
    ----------
    [1] Abdi, H (2007), « `Partial Least Square Regression`_ »

    [2] Garson, « `Partial Least Squares Regression (PLS)`_ » 

    [3] M. Tenenhaus (1998), « La régression PLS - Théorie et Pratique », Technip.

    [4] M. Tenenhaus (1995), « `Régression PLS et applications`_ », _Revue de statistique appliquée, tome 43, n°1, p. 7-63

    [5] R. Tomassone, M. Danzart, J.J. Daudin, J.P. Masson, « Discrimination et classement », Masson, 1988

    [6] Ricco Rakotomalala (2008), « `Analyse Discriminante PLS`_ », Université Lumière Lyon 2.

    [7] Ricco Rakotomalala (2008), « `Analyse Discriminante PLS - Etude comparative`_ », Université Lumière Lyon 2. 
    
    [8] Ricco Rakotomalala (2008), « `Régression PLS`_ », Université Lumière Lyon 2.

    [9] Ricco Rakotomalala (2008), « `Régression PLS - Sélection du nombre d'axes`_ », Université Lumière Lyon 2.

    [10] Ricco Rakotomalala (2008), « `Régression PLS - Comparaison de logiciels`_ », Université Lumière Lyon 2. 

    [11] Ricco Rakotomalala (2020), `Pratique de l'Analyse Discriminante Linéaire`_, Version 1.0, Université Lumière Lyon 2.
    
    [12] S. Chevallier, D. Bertrand, A. Kohler, P. Courcoux (2006), « `Application of PLS-DA in multivariate image analysis`_ », in J. Chemometrics, 20 : 221-229.
    
    [13] S. Vancolen (2004), « `La régression PLS`_ », Université de Neuchâtel.

    .. _Partial Least Square Regression: https://personal.utdallas.edu/~herve/Abdi-PLSR2007-pretty.pdf
    .. _Partial Least Squares Regression (PLS): http://www2.chass.ncsu.edu/garson/PA765/pls.htm
    .. _Régression PLS et applications: https://hal.science/hal-01604598v1/file/Rregression%20PLS_1.pdf
    .. _Analyse Discriminante PLS: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_PLS_DA.pdf
    .. _Analyse Discriminante PLS - Etude comparative: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_PLS_DA_Comparaison.pdf
    .. _Régression PLS: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_PLS.pdf
    .. _Régression PLS - Sélection du nombre d'axes: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_PLS_Selecting_Factors.pdf
    .. _Régression PLS - Comparaison de logiciels: https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_PLSR_Software_Comparison.pdf
    .. _Pratique de l'Analyse Discriminante Linéaire: https://hal.science/hal-04868585v1/file/Pratique_Analyse_Discriminante_Lineaire.pdf
    .. _Application of PLS-DA in multivariate image analysis: https://www.researchgate.net/publication/229902518_Application_of_PLS-DA_in_multivariate_image_analysis
    .. _La régression PLS: https://libra.unine.ch/handle/20.500.14713/28717

    Examples
    --------
    >>> from discrimintools.datasets import load_dataset
    >>> from discrimintools import CPLS
    >>> D = load_dataset("breast") # load traning data
    >>> y, X = D["Class"], D.drop(columns=["Class"]) # split into X and y
    >>> clf = CPLS()
    >>> clf.fit(X,y)
    CPLS()
    """
    def __init__(
           self, n_components = 2, scale = True, classes = None, max_iter = 500, tol = 1e-10, var_select = False, threshold = 1.0,  warn_message = True
    ):
        self.n_components = n_components
        self.scale = scale
        self.classes = classes
        self.max_iter = max_iter
        self.tol = tol
        self.var_select = var_select
        self.threshold = threshold
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
        C : Series of shape (n_samples,)
            Decision function values related to positive class.
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

        #check if X contains original features
        if not set(self.call_.Xtot.columns).issubset(X.columns):
            raise ValueError("The names of the features is not the same as the ones in the active features of the CPLS result")
        
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
        Xcod = Xcod.loc[:,list(self.call_.center.index)]
        #multiply by linear discriminant analysis coefficients
        return Xcod.dot(self.coef_.iloc[1:]).add(self.coef_.iloc[0])

    def fit(self,X,y):
        """
        Fit Partial Least Squares for Classification Model

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        y : Series of shape (n_samples,)
            Target values. True labels for ``X``. y must be a binary target

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

        #check if binary
        if len(list(y.unique())) != 2:
            raise TypeError("y must be a binary target.")

        #check if len are equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X must be equal to the number of samples in y")
        
        #check if all elements in y are string
        if not all(isinstance(kq, str) for kq in y):
            raise TypeError("All elements in y must be a string")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if max_iter is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.max_iter is None:
            max_iter = 500
        elif not isinstance(self.max_iter,(int,float)):
            raise TypeError("{} is not supported".format(type(self.max_iter)))
        elif self.max_iter < 0:
            raise ValueError("max_iter' must be positive")
        else:
            max_iter = self.max_iter
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if tol is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.tol is None:
            tol = 1e-10
        elif not isinstance(self.tol,(int,float)):
            raise TypeError("{} is not supported".format(type(self.tol)))
        elif self.tol < 0 or self.tol > 1:
            raise ValueError("the 'tol' value {} is not within the required range of 0 and 1.".format(self.tol))
        else:
            tol = self.tol

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if var_select is a bool
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_bool(self.var_select)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if threshold is not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.threshold is None:
            threshold = 1
        elif not isinstance(self.threshold,(int,float)):
            raise TypeError("{} is not supported".format(type(self.threshold)))
        elif self.threshold < 0 :
            raise ValueError("the 'threshold' value must be positive.")
        else:
            threshold = self.threshold

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

        #set y name if None
        if y.name is None or isinstance(y.name, int):
            y.name = "group"

        #warning message to inform
        if self.warn_message:
            if any(is_string_dtype(X[k]) for k in X.columns):
                print("\nCategorical features have been encoded into binary variables.\n")

        #recode to dummy if categorics variables
        X = model_matrix(X=X)

        #unique element in y
        uq_y = sorted(y.unique().tolist())
        #number of classes
        n_classes = len(uq_y)

        #class of categories
        if self.classes is not None and isinstance(self.classes, (list,tuple)):
            if len(list(set(self.classes) & set(uq_y))) != n_classes:
                raise ValueError("Insert good classes")
            classes = [str(x) for x in self.classes]
        else:
            classes = uq_y

        #convert y to categorical data type
        y = y.astype(CategoricalDtype(categories=classes,ordered=True))
        #set target name
        target = y.name
        #set piors
        priors = Series(array(y.value_counts(normalize=True).loc[classes]),index=classes,name="priors")
        #number of samples and features
        n_samples, n_features, features = X.shape[0], X.shape[1], list(X.columns)

        #recode
        Z = get_dummies(y,drop_first=True,dtype=int).iloc[:,0].map({0 : - priors.iloc[1], 1 : priors.iloc[0]})

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partial least squares regression (PLSR)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #QR decomposition of X
        Q, R = linalg.qr(X)
        #maximum number of components
        max_components = int(min(linalg.matrix_rank(Q), linalg.matrix_rank(R), n_samples - 1, n_features))
        #set number of components
        if self.n_components is None:
            n_components = max_components
        elif not isinstance(self.n_components,int):
            raise TypeError("'n_components' must be an integer.")
        elif self.n_components < 1:
            raise ValueError("'n_components' must be equal or greater than 1.")
        else:
            n_components = min(self.n_components, max_components)

        #partial least squares regression
        plsr = PLSRegression(n_components=n_components,scale=self.scale,max_iter=max_iter,tol=tol).fit(X,Z)
        #variable importance in projection
        vip = plsrvip(obj=plsr,threshold=threshold)

        #if selected variables
        if self.var_select:
            #update X
            X = X[vip.selected]
            #update partial least square regression
            plsr = PLSRegression(n_components=n_components,scale=self.scale,max_iter=max_iter,tol=tol).fit(X,Z)
            #update variable importance in projection
            vip = plsrvip(obj=plsr,threshold=threshold)

        #update number of features and features in
        n_features, features = plsr.n_features_in_, plsr.feature_names_in_

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #convert to pandas DataFrame
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #center and scale
        x_center, x_scale = Series(plsr._x_mean,index=plsr.feature_names_in_,name="center"), Series(plsr._x_std,index=plsr.feature_names_in_,name="scale")
        
        #convert to ordered dictionary
        call_ = OrderedDict(Xtot=Xtot,X=X,y=y,target=target,features=features,classes=classes,priors=priors,center=x_center,scale=x_scale,
                            n_samples=n_samples,n_features=n_features,n_classes=n_classes,max_components=max_components,n_components=n_components,
                            max_iter=max_iter,tol=tol,threshold=threshold)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables importance in projection
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.vip_ = vip

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coefficients of partial least squares regression
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #coefficients of positive category
        coef_ = Series(plsr.coef_[0],index=plsr.feature_names_in_,name=classes[1])
        #intercept
        intercept_ = Series(plsr._y_mean - dot(x_center,coef_),index=["Constant"],name=classes[1])
        #coefficients of partial least-squares regression
        self.coef_ = concat((intercept_,coef_),axis=0)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations : coordinates and scores
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = DataFrame(plsr.x_scores_,index=X.index,columns=["Can{}".format(x+1) for x in range(n_components)])
        #individuals scores - decision function
        ind_scores = X.dot(self.coef_.iloc[1:]).add(self.coef_.iloc[0])
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #classes informations: coordinates, squared euclidean distance and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #count and proportion
        n_k, p_k = y.value_counts(normalize=False).loc[classes],  y.value_counts(normalize=True).loc[classes]
        #class level information
        class_infos = DataFrame(c_[n_k,p_k,priors],columns=["Frequency","Proportion","Prior Probability"],index=classes)
        class_infos["Frequency"] = class_infos["Frequency"].astype(int)
        #classes coordinates
        class_coord = concat((ind_coord.loc[y[y==k].index,:].mean(axis=0).to_frame(k) for k in classes),axis=1).T
        #squared euclidean distance between classes
        class_eucl = DataFrame(squareform(pdist(class_coord,metric="sqeuclidean")),index=classes,columns=classes)
        #squared generalized distance
        class_gen = class_eucl.sub(2*log(priors),axis=1)
        #convert to ordered dictionary
        classes_ = OrderedDict(infos=class_infos,coord=class_coord,eucl=class_eucl,gen=class_gen)
        #convert to namedtuple
        self.classes_ = namedtuple("classes",classes_.keys())(*classes_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals additional informations: squared euclidean distance between classes barycenters and squared generalized distance
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #squared euclidean distance to class center
        ind_eucl = concat((ind_coord.sub(class_coord.loc[k,:],axis=1).pow(2).sum(axis=1).to_frame(k) for k in classes),axis=1)
        #squared generalized distance
        ind_gen = ind_eucl.sub(2*log(priors),axis=1)
        #convert to ordered dictionary
        ind_ = OrderedDict(coord=ind_coord,scores=ind_scores,eucl=ind_eucl,gen=ind_gen)
        #convert to namedtuple
        self.ind_ = namedtuple("ind",ind_.keys())(*ind_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables informations : weights, loadings and rotations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables weights
        var_weights = DataFrame(plsr.x_weights_,index=features,columns=ind_coord.columns)
        #variables loadings
        var_loadings = DataFrame(plsr.x_loadings_,index=features,columns=var_weights.columns)
        #variables rotations
        var_rotations = DataFrame(plsr.x_rotations_,index=features,columns=var_weights.columns)
        #convert to ordered dictionary
        var_ = OrderedDict(weights=var_weights,loadings=var_loadings,rotations=var_rotations)
        #convert to namedtuple
        self.var_  = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #explained variance for X by each components 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total variance in Z
        total_var = X.sub(x_center,axis=1).div(x_scale,axis=1).var(axis=0,ddof=1).sum()
        #explained variance
        explained_var = 100*(ind_coord.var(axis=0,ddof=1)*var_loadings.pow(2).sum(axis=0))/total_var
        #convert to DataFrame
        self.explained_variance_ = DataFrame(c_[explained_var,cumsum(explained_var)],columns=["Proportion (%)","Cumulative (%)"],index=explained_var.index)
        
        #set model name
        self.model_ = "cpls"
        return self

    def fit_transform(self,X,y):
        """
        Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        y : Series of shape (n_samples, n_targets)
            Target values. True labels for ``X``. 
        
        Returns
        -------
        X_new : DataFrame of shape (n_samples, n_components)
            The transformed training samples, where ``n_components`` is the number of pls components.
        """
        self.fit(X,y)
        return self.ind_.coord
    
    def inverse_transform(self,X):
        """
        Transform data back to its original space.

        This transformation will only be exact if ``n_components=n_features``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_components)
            New data, where ``n_samples`` is the number of samples and ``n_components`` is the number of pls components.

        Returns
        -------
        X_original : DataFrame of shape (n_samples, n_features)
            Return the reconstructed ``X`` data, where ``n_features`` is the number of features.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if the estimator is fitted by verifying the presence of fitted attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_is_fitted(self)

        #from pls space to original space and enormalize
        return X.dot(self.var_.loadings.T).mul(self.call_.scale,axis=1).add(self.call_.center,axis=1)
    
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
        #predicted labels for X
        return Series(array(self.call_.classes)[where(self.decision_function(X) >= 0,1,0)],index=X.index,name="prediction")
    
    def transform(self,X):
        """
        Apply the dimension reduction.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        x_scores : DataFrame of shape (n_samples, n_components)
            The transformed input data, where ``n_components`` is the number of pls components.
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

        #check if X contains original features
        if not set(self.call_.Xtot.columns).issubset(X.columns):
            raise ValueError("The names of the columns is not the same as the ones in the active features of the CPLS result")
        
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
        Xcod = Xcod.loc[:,list(self.call_.center.index)]
        #standardize : Z = (X - center)/scale and apply rotation
        return Xcod.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).dot(self.var_.rotations)