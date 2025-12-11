******
PLSLDA
******

Overview
--------

:class:`~discrimintools.discriminant_analysis.PLSLDA` combines PLS Regression and Linear Discriminant Analysis. It can handle multiclass
problem i.e. the target variable can have :math:`K \left(K \geq 2\right)` classes. It relies on the same principle
(as :class:`~discrimintools.discriminant_analysis.CPLS` and :class:`~discrimintools.discriminant_analysis.PLSDA`) about the number of components detection.

Description of the method
-------------------------

There are two main steps in the learning process. Firstly, using the same coding scheme as :class:`~discrimintools.discriminant_analysis.PLSDA`,
we launch the PLS algorithm. Secondly, we launch the linear discriminant analysis on the X
component scores (factors). Because these factors are orthogonal, the :class:`~discrimintools.discriminant_analysis.DISCRIM` is more reliable. This
kind of data transformation is very useful when the original input variables are highly correlated.

Predictive idea
---------------

The classification rule is the same as the :class:`~discrimintools.discriminant_analysis.PLSDA` component.

.. note::
    Because we combine two approaches, it is very difficult to evaluate the influence of the 
    original features. For this reason no information is provided about the relevance of features, but about the pls components.

Number of components
--------------------

In :class:`~discrimintools.discriminant_analysis.PLSLDA` procedure, we can explicitly specify the number of components, with the parameter ``n_components``, for NIPALS [1]_ algorithms.

.. rubric:: References

.. [1] NonLinear Iterative Partial Least Squares.