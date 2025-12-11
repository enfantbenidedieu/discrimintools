*****
PLSDA
*****

Overview
---------

`Partial Least Squares <https://en.wikipedia.org/wiki/Partial_least_squares_regression>`_ can also be applied to classification problems. The general idea is to perform a PLS(2) decomposition between :math:`X` and :math:`Z` 
where :math:`Z` is `dummy encoded <https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html>`_ for the different classes. The scores that come from the PLS decomposition are then used as an input to a classification model. 
So really, PLSDA is just using PLS to find a good subspace then performing classication on the transformed coordinates in that space; this classification is the “discrimination” which can be done by any number of methods.

Description of the method
-------------------------

Partial least squares discriminant analysis (PLSDA) can handle multiclass problem i.. the target variable can have :math:`K` (:math:`K \geq 2`) classes.
It is relies on the same principle as :class:`~discrimintools.discriminant_analysis.CPLS` about the dectection of the number of components. 

In this approach, we create :math:`K` indicator variable (as much as the number of classes) using the following coding scheme:

.. math::
    Z_{k} = \begin{cases} 1 & \text{if}\quad y_{k}=k \\  0 & \text{otherwise}\end{cases}

The PLS algorithm handles the :math:`Z` target variables and the :math:`X` features. We obtain :math:`K` classification functions:

.. math::
    d\left(y_{k},X\right) = \beta_{k}^{T}X = \beta_{0,k} + \beta_{1,k}X_{1} + \cdots + \beta_{k,p}X_{p}

Predictive idea
---------------

The classification rule used in :class:`~discrimintools.discriminant_analysis.PLSDA` consists of assigning each individual :math:`i` to the class :math:`\mathcal{C}_{k}` using the following rule :

.. math::
    \widehat{y}_{k} = \text{arg}\underbrace{\max}_{l}\left\{d\left(y_{l},X\right)\right\}

Number of components
--------------------

In :class:`~discrimintools.discriminant_analysis.PLSDA` procedure, we can explicitly specify the number of components, with the parameter ``n_components``, for NIPALS [1]_ algorithms.

VIP
---

You can use VIP (variable importance in the projection) to select predictor variables when multicollinearity exists among variables. The VIP coefficients reflects the relative 
importance for the selected factors. 

Description
^^^^^^^^^^^

The VIP for a feature :math:`j` in PLSDA model with :math:`H` components is given as:

.. math::
    VIP_{j} = \sqrt{\dfrac{p}{\displaystyle \sum_{h=1}^{h=H}R^{2}\left(y,t_{h}\right)}\displaystyle \sum_{h=1}^{h=H}R^{2}\left(y,t_{h}\right) w_{j,h}^{2}}

where :math:`R^{2}\left(y,t_{h}\right)` is the square correlation coefficient between :math:`y` and :math:`t_{h}`; :math:`w_{j,h}` is the :math:`x`-weight coefficient.

Variables with a VIP score greater than :math:`1` (default ``threshold`` in :class:`~discrimintools.discriminant_analysis.PLSDA` procedure) are considered important for the projection of the PLS regression.

.. note::
    These selections rules must be use with caution because the VIP reflects only the relative importance (each others) of the input variables. 
    It does not mean that a variable with a low VIP is not relevant for the classification.

Coefficients
------------
Coefficients are the parameters in a regression equation. The estimated coefficients are used with the predictors to calculate the fitted value of the response variable and the predicted response of new observations. 
In contrast to least squares, the PLS coefficients are nonlinear estimators. 
Standardized coefficients indicate the importance of each predictor in the model and correspond to the standardized :math:`x`- and :math:`z`-variables. 
In PLS, the coefficient matrix of shape :math:`(p,K)` is calculated from the weights and loadings.

The formula for standardized coefficients is:

.. math::
    \beta^{std} = W\left(P^{T}W\right)^{-1}Q^{T}

To calculate the nonstandardized coefficients and intercept, use these formulas:

.. math::
    \beta_{j,k} & = \beta_{j,k}^{std} \dfrac{\sigma_{Z_{k}}}{\sigma_{j}} \\
     \beta_{0,k} & = \mu_{Z_{k}} - \displaystyle \sum_{j} \mu_{j}\beta_{j,k}

where:

==========  ======================================================
Terms       Description
==========  ======================================================
:math:`W`   the :math:`x`-weight matrix
:math:`P`   the :math:`x`-loading matrix
:math:`Q`   the :math:`Z`-loading matrix
:math:`j`   the features :math:`j`
:math:`p`   the number of features
:math:`K`   the number of classes (targets)
==========  ======================================================

Explained variance of :math:`X`
-------------------------------

The explained variance ratio is defined by the following formula:

.. math::
    \text{Explained variance ratio} = \dfrac{\text{Variance explained by component}}{\text{Total variance}}

which equal to:

.. math::
    \text{Explained variance ratio}(h) = \dfrac{\lvert \lvert t_{h}p_{h}^{T}\rvert \rvert_{F}^{2}}{\lvert \lvert X\rvert \rvert_{F}^{2}}

where:

* :math:`\lvert \vert \cdot \rvert \rvert_{F}` is the Frobenius norm (sum of squres of all entries)
* :math:`\lvert \vert X \rvert \rvert_{F}^{2}=\displaystyle \sum_{i}\sum_{j} X_{ij}^{2}=` total variance of :math:`X` (centered or standardized).

.. rubric:: References

.. [1] NonLinear Iterative Partial Least Squares