.. _getting_started:

===============
Getting started
===============

`discrimintools <https://github.com/enfantbenidedieu/discrimintools>`_ is an open source python library dedicated to Discriminant Analysis (DA) distributed under the MIT Licence.

The purpose of this guide is to illustrate some of the main features of ``discrimintools``. It assumes basic working knowledge of `scikit-learn <https://scikit-learn.org/stable/>`_ practices.

Fitting and predicting: estimator basics
----------------------------------------

As `scikit-learn <https://scikit-learn.org/stable/index.html#>`_, `discrimintools <https://github.com/enfantbenidedieu/discrimintools>`_ provides models called `estimators <https://scikit-learn.org/stable/glossary.html#term-estimators>`_. Each estimator can be fitted to some data using its `fit <https://scikit-learn.org/stable/glossary.html#term-fit>`_ method.

Here is a simple example where we fit a :class:`~discrimintools.discriminant_analysis.DISCRIM` to :class:`~discrimintools.datasets.load_alcools` data:

    >>> from discrimintools.datasets import load_alcools
    >>> from discrimintools import DISCRIM
    >>> D = load_alcools("train") # load training data
    >>> y, X = D["TYPE"], D.drop(columns=["TYPE"]) # split into X and y
    >>> clf = DISCRIM()
    >>> clf.fit(X, y)
    DISCRIM(priors='prop')
  
The ``fit`` method generally accepts 2 inputs:

- The samples DataFrame (or design matrix) ``X``. The size of ``X``
  is typically ``(n_samples, n_features)``, which means that samples are
  represented as rows and features are represented as columns.
- The target values ``y`` which are true lables for ``X``. ``y`` is a pandas Series with categorical terms. For
  unsupervised learning tasks, ``y`` does not need to be specified, such as :class:`~discrimintools.discriminant_analysis.GFA` and :class:`~discrimintools.discriminant_analysis.MPCA`.

Both ``X`` and ``y`` are expected to be pandas objects : `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ for ``X`` and `Categorical Series <https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_ for ``y``.

Once the estimator is fitted, it can be used for predicting target values of
new data. You don't need to re-train the estimator::

    >>> DTest = load_alcools("test") # load testing data
    >>> yTest, XTest = DTest["TYPE"], DTest.drop(columns=["TYPE"])
    >>> clf.predict(X).head()  # predict classes of the training data
    ...
    >>> clf.predict(XTest).head()  # predict classes of new (test) data 
    
Pipelines: chaining pre-processors and estimators
-------------------------------------------------

Transformers and estimators (predictors) can be combined together into a
single unifying object: a `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_. The sklearn pipeline
offers the same API as a regular estimator: it can be fitted and used for
prediction with ``fit`` and ``predict``. As we will see later, using a
pipeline will also prevent you from data leakage, i.e. disclosing some
testing data in your training data.

In the following example, we use the :class:`~discrimintools.datasets.load_alcools` dataset to run a linear discriminant analysis on principal components:

    >>> from discrimintools.datasets import load_alcools
    >>> from sklearn.pipeline import Pipeline
    >>> from discrimintools import GFA, DISCRIM
    >>> D = load_alcools("train")
    >>> y, X = D["TYPE"], D.drop(columns=["TYPE"]) # split into X and y
    ...
    >>> #create a pipeline object
    >>> clf = Pipeline([("gfa",GFA(n_components=2)),
    ...                 ("lda",DISCRIM(method="linear",warn_message=False))])
    >>> clf.fit(X,y)

For more about machine learning terms, see `sklearn glossary <https://scikit-learn.org/stable/glossary.html>`_.