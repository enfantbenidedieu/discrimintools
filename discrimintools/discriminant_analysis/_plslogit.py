# -*- coding: utf-8 -*-
from numpy import array, linalg, c_, cumsum, log, where
from pandas import DataFrame, CategoricalDtype, get_dummies, Series, concat
from pandas.api.types import is_string_dtype
from collections import OrderedDict, namedtuple
from sklearn.cross_decomposition import PLSRegression
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
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

class PLSLOGIT(_BaseDA):
    """
    Partial Least Squares Logistic Regression (PLSLOGIT)

    Performs partial least squares logistic regression (PLSLOGIT). It's a classical logistic regression (binary, multinomial, ordinal) carried out on the scores of a partial least scores of explanatory variables.
    Partial least squares logistic regression consists in three steps:
    
    1. Recode the target variable into ``n_classes`` dummy variables.
    2. Computation of partial least squares regression using `PLSRegression <https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html>`_.
    3. Computation of logistic regression (binary, multinomial, ordinal) on ``x_scores`` extract in step 2 using `Statsmodels <https://www.statsmodels.org/v0.14.4/index.html>`_.   

    Parameters
    ----------
    n_components : int or None, default = 2
        Number of components to keep. Should be in ``[1, n_features]``.

    scale : bool, defaul = True
        Whether to scale ``X`` and ``y``.

    classes : None, tuple or list, default = None
        Name of level in order to return. If ``None``, classes are sorted using unique values in y.

    max_iter : int, default = 500
        The maximum number of iterations for NIPALS method
    
    tol : float, default = 1e-06
        The tolerance used as convergence criteria in the NIPALS method.

    var_select : bool, default = True
        Whether to applied feature selection based on variables importance in Projection for Partial Least-Squares Regression

    threshold : float, default = 1.0
        You can use VIP to select predictor variables when multicollinearity exists among variables. 
        Variables with a VIP score greater than 1 are considered important for the projection of the PLS regression.

    multi_class : None, str.
        You can choose between ``multinomial`` or ``ordinal`` logistic regression. Only for multiclass problem.

    warn_message : bool, default = True
        Whether to show warning messages.

    kwargs : 
        Additionals parameters to used in ``fit`` for logistic regression. see `statsmodels <https://www.statsmodels.org/v0.14.4/index.html>`_.

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
            Names of classes
        - priors : Series of shape (n_classes,)
            Priors probabilities
        - center : Series of shape (n_features,)
            The average of `X`
        - scale : Series of shape (n_features,)
            The standard deviation of ``X``.
        - n_samples : int
            Number of samples.
        - n_features : int
            Number of features.
        - max_components : int
            Maximum number of components.
        - n_components : int
            Number of components kept.
        - n_classes : int
            Number of target values.
        - max_iter : int
            Maximum number of iterations.
        - tol : float
            The tolerance used as convergence criteria.
        - threshold : float,
            The tolerance for variable importance in projection.
        - multi_class : None, str
            The multiclass logistic regression applied.

    cancoef_ : NamedTuple
        Canonical coefficients:

        - standardized : DataFrame of shape (n_variables, n_components)
            The standardized canonical coefficients
        - raw : DataFrame of shape (n_variables + 1, n_components)
            The raw canonical coefficients

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

    coef_ : NamedTuple
        Partial least squares logit model coefficients:

        - standardized : DataFrame of shape (n_variables, n_classes - 1)
            The standardized coefficients.
        - raw : DataFrame of shape (n_variables+1, n_classes - 1)
            The raw coefficients.

    explained_variance_ : DataFrame of shape (n_components, 2)
        The explained variance and the cumulative explained variance.

    ind_ : NamedTuple
        Individuals informations:

        - coord : DataFrame of shape (n_samples, n_components)
            The transformed training simples.
        - scores : DataFrame of shape (n_samples,) or (n_samples, n_classes - 1)
            The total scores of individuals.
        - eucl : DataFrame of shape (n_samples, n_classes)
            The squared Euclidean distance to origin.
        - gen : DataFrame shape (n_samples, n_classes) 
            The generalized squared distance to origin.

    logit_ : class
        An object of class Logit.

    logit_coef_ : DataFrame of shape (n_components + 1,) or (n_components + 1, n_classes - 1)
        Logistic regression model coefficients.

    model_ : str, default = 'plslogit'
        The model fitted name.

    var_ : NamedTuple
        Variables informations:

        - weights : DataFrame of shape (n_features, n_components)
            The left singular vectors of the cross-covariance matrices of each iteration.

        - loadings : DataFrame of shape (n_features, n_components)
            The loadings of `X`.

        - rotations : DataFrame of shape (n_features, n_components)
            The projection matrix used to transform X.

    See also
    --------
    :class:`~discrimintools.discriminant_analysis.PLSDA`
        Partial Least Squares Linear Discriminant Analysis
    :class:`~discrimintools.summary.summaryPLSLOGIT`
        Printing summaries of Partial Least Squares Linear Logistic Regression model.
    :class:`~discrimintools.summary.summaryDA`
        Printing summaries of Discriminant Analysis model.

    References
    ----------
    [1] Droesbeke J. J., Lejeune M., Saporta G.  (2005), « `Modèles statistiques pour données qualitatives <https://www.editionstechnip.com/fr/catalogue-detail/995/modeles-statistiques-pour-donnees-qualitatives.html>`_ », Editions TECHNIP.

    [2] Tuffery S. (2017), « Data Mining et Statistique décisionnelle : La science des données », Editions TECHNIP.

    [3] Tuffery S. (2024), « `Modélisation prédictive et Apprentissage statistique avec R <https://www.editionstechnip.com/fr/catalogue-detail/2145/modelisation-predictive-et-apprentissage-statistique-avec-r.html>`_ », Editions TECHNIP, 5ed;

    [4] Tuffery R. (2025), « `Data Science, Statistique et Machine Learning <https://www.editionstechnip.com/fr/catalogue-detail/1005/data-science-statistique-et-machine-learning.html>`_ », Editions TECHNIP, 6ed.

    Examples
    --------
    >>> from discrimintools.datasets import load_dataset, load_vins
    >>> from discrimintools import PLSLOGIT
    >>> #pls + logit
    >>> D = load_dataset("breast")
    >>> y, X = D["Class"], D.drop(columns=["Class"])
    >>> clf = PLSLOGIT()
    >>> clf.fit(X,y)
    PLSLOGIT()
    >>> D = load_vins("train")
    >>> y, X = D["Qualite"], D.drop(columns=["Qualite"])
    >>> #pls + multinomial
    >>> clf = PLSLOGIT(classes=('Mediocre','Moyen','Bon'))
    >>> clf.fit(X,y)
    PLSLOGIT(classes=('Mediocre','Moyen','Bon'))
    >>> "pls + ordinal
    >>> clf = PLSLOGIT(multi_class="ordinal",classes=('Mediocre','Moyen','Bon'),method='bfgs')
    >>> clf.fit(X,y)
    PLSLOGIT(multi_class="ordinal",classes=('Mediocre','Moyen','Bon'),method='bfgs')
    """
    def __init__(
            self, n_components = 2, scale = True, classes = None, max_iter = 500, tol = 1e-10, var_select = False, threshold = 1.0, multi_class=None, warn_message = True, **kwargs
    ):
        self.n_components = n_components
        self.scale = scale
        self.classes = classes
        self.max_iter = max_iter
        self.tol = tol
        self.var_select = var_select
        self.threshold = threshold
        self.multi_class = multi_class
        self.warn_message = warn_message
        self.kwargs = kwargs

    def decision_function(self,X):
        """
        Apply decision function to an input data

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        C : DataFrame of shape (n_samples, ) or (n_samples, n_classes - 1)
            Decision function values
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
            raise ValueError("The names of the features is not the same as the ones in the active features of the PLSLOGIT result")
        
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
        #multiply by coefficients
        if self.call_.n_classes == 2:
            C = Xcod.dot(self.coef_.raw.iloc[1:]).add(self.coef_.raw.iloc[0])
        else:
            C = Xcod.dot(self.coef_.raw.iloc[1:,:]).add(self.coef_.raw.iloc[0,:].values,axis=1)
        return C

    def fit(self,X,y):
        """
        Fit Partial Least Squares Logistic Regression Model

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        
        y : Series of shape (n_samples,)
            Target values. True values for ``X``.

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

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #check if multi_class is assigned for multiclass logistic regression
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if n_classes > 2:
            if self.multi_class is None:
                multi_class = "multinomial"
            elif self.multi_class not in ["multinomial","ordinal"]:
                raise ValueError("'multi_class' should be one of 'multinomial', 'ordinal'.")
            else:
                multi_class = self.multi_class
        else:
            multi_class = None

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

        #create disjunctive table
        Y = get_dummies(y,dtype=int)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #partial least square regression (PLSR)
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
        plsr = PLSRegression(n_components=n_components,scale=self.scale,max_iter=max_iter,tol=tol).fit(X,Y)
        #variable importance in projection
        vip = plsrvip(obj=plsr,threshold=threshold)

        #if selected variables
        if self.var_select:
            #update X
            X = X[vip.selected]
            #update partial least squares regression
            plsr = PLSRegression(n_components=n_components,scale=self.scale,max_iter=max_iter,tol=tol).fit(X,Y)
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
                            n_samples=n_samples,n_features=n_features,max_components=max_components,n_components=n_components,n_classes=n_classes,
                            max_iter=max_iter,tol=tol,threshold=threshold,multi_class=multi_class)
        #convert to namedtuple
        self.call_ = namedtuple("call",call_.keys())(*call_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables informations : weights, loadings and rotations
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #variables weights
        var_weights = DataFrame(plsr.x_weights_,index=features,columns=["Can{}".format(x+1) for x in range(n_components)])
        #variables loadings
        var_loadings = DataFrame(plsr.x_loadings_,index=features,columns=var_weights.columns)
        #variables rotations
        var_rotations = DataFrame(plsr.x_rotations_,index=features,columns=var_weights.columns)
        #convert to ordered dictionary
        var_ = OrderedDict(weights=var_weights,loadings=var_loadings,rotations=var_rotations)
        #convert to namedtuple
        self.var_  = namedtuple("var",var_.keys())(*var_.values())

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #canonical discriminant coefficients
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total-sample standardized canonical coefficients
        std_cancoef = var_loadings
        #total-sample raw (unstandardized) canonical coefficients
        raw_cancoef = concat((std_cancoef.T.dot(-x_center/x_scale).to_frame("Constant").T, std_cancoef.div(x_scale,axis=0)),axis=0)
        #convert to ordered dictionary
        cancoef_ = OrderedDict(standardized=std_cancoef,raw=raw_cancoef)
        #convert to namedtuple
        self.cancoef_ = namedtuple("cancoef",cancoef_.keys())(*cancoef_.values())
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals informations : coordinates and scores
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #individuals coordinates
        ind_coord = DataFrame(plsr.x_scores_,index=X.index,columns=var_weights.columns)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #logistic regression
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #add constant to PLS components
        x_coord = sm.add_constant(ind_coord)
        if n_classes == 2:
            #binary logistic regression
            clf = sm.Logit(y.cat.codes,x_coord).fit(disp=self.warn_message,**self.kwargs)

            #binary logistic coefficients
            clf_coef = clf.params
            #set name
            clf_coef.name = classes[1]
            
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #classification functions
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #standardize coefficients
            std_coef = concat((clf.params.iloc[:1],std_cancoef.dot(clf.params.iloc[1:])),axis=0)
            #concatenate
            raw_coef = raw_cancoef.dot(clf.params.iloc[1:])
            #update constante
            raw_coef.update(raw_coef.iloc[:1].add(clf.params.iloc[0]))
            #set name
            std_coef.name, raw_coef.name = classes[1], classes[1]
            #convert to ordered dictionary
            coef_ = OrderedDict(standardized=std_coef,raw=raw_coef)
            #convert to namedtuple
            self.coef_ = namedtuple("coef",coef_.keys())(*coef_.values())

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #individuals scores - decision function
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            ind_scores = X.dot(raw_coef.iloc[1:]).add(raw_coef.iloc[0])
        else:
            if multi_class == "multinomial":
                #multinomial logistic regression
                clf = sm.MNLogit(y,x_coord).fit(disp=self.warn_message,**self.kwargs)
                #coefficients of multinomial logistic regression
                clf_coef = clf.params
                clf_coef.columns = classes[1:]
            else:
                #cumulative logistic regression
                clf = OrderedModel(y,ind_coord,distr='logit').fit(disp=self.warn_message,**self.kwargs)

                #coefficients of cumulative logistic regression
                clf_coef, clf_cst = clf.params.iloc[:n_components], clf.params.iloc[n_components:]
                clf_cst.name = "Constant"
                #replicate coeffiicients
                clf_coef = concat((clf_coef for x in clf_cst.index),axis=1)
                clf_coef.columns = clf_cst.index
                #concatenate
                clf_coef = concat((clf_cst.to_frame().T,clf_coef),axis=0)
            
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #classification functions
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #standardize coefficients
            std_coef = concat((clf_coef.iloc[0,:].to_frame().T,std_cancoef.dot(clf_coef.iloc[1:,:])),axis=0)
            #concatenate
            raw_coef = raw_cancoef.iloc[1:,:].dot(clf_coef.iloc[1:,:])
            #update constante
            raw_cst = raw_cancoef.iloc[0,:].to_frame().T.dot(clf_coef.iloc[1:,:]).add(clf_coef.iloc[0,:].values,axis=1)
            #concatenate
            raw_coef = concat((raw_cst,raw_coef),axis=0)
            #convert to ordered dictionary
            coef_ = OrderedDict(standardized=std_coef,raw=raw_coef)
            #convert to namedtuple
            self.coef_ = namedtuple("coef",coef_.keys())(*coef_.values())

            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #individuals scores - decision functions
            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            ind_scores = X.dot(raw_coef.iloc[1:,:]).add(raw_coef.iloc[0,:],axis=1)
            
        #store logistic informations
        self.logit_, self.logit_coef_ = clf, clf_coef

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
        #explained variance for X by each components 
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #total variance in Z
        total_var = X.sub(x_center,axis=1).div(x_scale,axis=1).var(axis=0,ddof=1).sum()
        #explained variance
        explained_var = 100*(ind_coord.var(axis=0,ddof=1)*var_loadings.pow(2).sum(axis=0))/total_var
        #convert to DataFrame
        self.explained_variance_ = DataFrame(c_[explained_var,cumsum(explained_var)],columns=["Proportion (%)","Cumulative (%)"],index=explained_var.index)

        #set model name
        self.model_ = "plslogit"
        return self
    
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
        #estimated probabilities
        y_prob = self.predict_proba(X)
        if self.call_.n_classes == 2:
            y_pred = Series(array(self.call_.classes)[where(y_prob >= 0.5,1,0)],index=X.index,name="prediction")
        else:
            y_pred = y_prob.idxmax(axis=1)
            y_pred.name = "prediction"
        return y_pred
    
    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        C : DataFrame of shape (n_samples,) or (n_samples, n_classes)
            Posterior log-probabilities
        """
        #estimated log-probabilities
        return self.predict_proba(X).transform(log)
    
    def predict_proba(self,X):
        """
        Probability estimates.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            The data for which we want to get the predictions.
        
        Returns
        -------
        C : DataFrame of shape (n_samples,) or (n_samples, n_classes)
            Estimated probabilities.
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
            raise ValueError("The names of the features is not the same as the ones in the active features of the PLSLOGIT result")
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
        coord = Xcod.sub(self.call_.center,axis=1).div(self.call_.scale,axis=1).dot(self.var_.rotations)
        #add constant
        if not (self.call_.n_classes > 2 and self.call_.multi_class == "ordinal"):
            coord = sm.add_constant(coord)
        #predicted probabilities
        y_prob = self.logit_.predict(coord)
        #set columns in case of multi class
        if self.call_.n_classes > 2:
            y_prob.columns = self.call_.classes
        return y_prob