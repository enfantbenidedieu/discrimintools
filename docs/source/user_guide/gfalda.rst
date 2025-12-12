******
GFALDA
******

Overview
--------

:class:`~discrimintools.discriminant_analysis.GFALDA` combines General factor analysis :class:`~discrimintools.discriminant_analysis.GFA` and Linear Discriminant Analysis. It can handle multiclass
problem i.e. the target variable can have :math:`K \left(K \geq 2\right)` classes.

Description of the method
-------------------------

There are two main steps in the learning process. Firstly, we launch the general factor analysis algorithm on :math:`X`. 
Secondly, we launch the linear discriminant analysis on the X component scores (factors). 
Because these factors are orthogonal, the :class:`~discrimintools.discriminant_analysis.DISCRIM` is more reliable. This
kind of data transformation is very useful when the original input variables are highly correlated.

.. note::
    1. If all features are numerics, then GFALDA corresponds to principal component analysis discriminant analysis (PCADA)
    2. If all features are categorics, then GFALDA corresponds to discriminant analysis on qualitative predictors (`DISQUAL <https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Pipeline_Python.pdf>`_)
    3. If mixed features, then GFALDA corresponds to discriminant analysis on mixed predictors (DISMIX)

Predictive idea
---------------

The classification rule is the same as the :class:`~discrimintools.discriminant_analysis.DISCRIM` component.

.. note::
    Because we combine two approaches, it is very difficult to evaluate the influence of the 
    original features. For this reason no information is provided about the relevance of features, but about the components.

Number of components
--------------------

In :class:`~discrimintools.discriminant_analysis.GFALDA` procedure, we can explicitly specify the number of components, with the parameter ``n_components``.