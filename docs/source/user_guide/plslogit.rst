********
PLSLOGIT
********

Overview
--------

:class:`~discrimintools.PLSLOGIT` combines PLS Regression and Logistic Regression. It can handle multiclass
problem i.e. the target variable can have :math:`K \left(K \geq 2\right)` classes. It relies on the same principle
:class:`~discrimintools.PLSLDA` about the number of components detection.

Description of the method
-------------------------

There are two main steps in the learning process. Firstly, using the same coding scheme as :class:`~discrimintools.PLSDA`,
we launch the PLS algorithm. Secondly, we launch the logistic regression on the X
component scores (factors). This kind of data transformation is very useful when the original input variables are highly correlated.

Predictive idea
---------------

The classification rule is the same as the logistic regression component.

Number of components
--------------------

In :class:`~discrimintools.PLSLOGIT` procedure, we can explicitly specify the number of components, with the parameter ``n_components``, for NIPALS [1]_ algorithms.

.. rubric:: References

.. [1] NonLinear Iterative Partial Least Squares.