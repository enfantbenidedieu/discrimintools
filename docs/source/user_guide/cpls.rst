****
CPLS
****

Overview
--------
Partial least squares for classification (CPLS) is dedicated to binary problem i.e the target variable must have two values only.

Description of the method
-------------------------

The target variable is replaced by a continuous variable using a specific code [1]_ and the `PLSR <https://fr.wikipedia.org/wiki/R%C3%A9gression_des_moindres_carr%C3%A9s_partiels>`_ is applied.

``y`` is the target variable, with :math:`y = \left\{+,-\right\}`. If :math:`n_{+}` (resp. :math:`n_{-}`) is the number of positive (resp. negative) in the sample and :math:`n = n_{+} + n_{-}`.
The variable :math:`Z` is defined as follows:

.. math::
    Z = \begin{cases} \dfrac{n_{-}}{n} & \text{if}\quad y = + \\ -\dfrac{n_{+}}{n} & \text{if}\quad y = -\end{cases}

This approach produces one discriminant function :math:`D(X)` such as:

.. math::
    D(X) = \beta^{T}X

Predictive idea
---------------

The classification rule used in CPLS consists of assigning each individual :math:`i` to the class :math:`\{+,-\}` using the following rule :

.. math::
    \widehat{y} = \begin{cases} + & \text{if} \quad D_{i}(X) \geq 0 \\ - & \text{if} \quad D_{i}(X) < 0\end{cases}

Number of components
--------------------

In :class:`~discrimintools.CPLS` procedure, we can explicitly specify the number of components, with the parameter ``n_components``, for NIPALS [2]_ algorithms.

VIP
---

You can use VIP (variable importance in the projection) to select predictor variables when multicollinearity exists among variables. The VIP coefficients reflects the relative 
importance for the selected factors. 

Description
^^^^^^^^^^^

The VIP for a feature :math:`j` in CPLS model with :math:`H` components is given as:

.. math::
    VIP_{j} = \sqrt{\dfrac{p}{\displaystyle \sum_{h=1}^{h=H}R^{2}\left(y,t_{h}\right)}\displaystyle \sum_{h=1}^{h=H}R^{2}\left(y,t_{h}\right) w_{j,h}^{2}}

where :math:`R^{2}\left(y,t_{h}\right)` is the square correlation coefficient between :math:`y` and :math:`t_{h}`; :math:`w_{j,h}` is the :math:`x`-weight coefficient.

Variables with a VIP score greater than :math:`1` (default ``threshold`` in :class:`~discrimintools.CPLS` procedure) are considered important for the projection of the PLS regression.

.. note::
    These selections rules must be use with caution because the VIP reflects only the relative importance (each others) of the input variables. 
    It does not mean that a variable with a low VIP is not relevant for the classification.

Coefficients
------------
Coefficients are the parameters in a regression equation. The estimated coefficients are used with the predictors to calculate the fitted value of the response variable and the predicted response of new observations. 
In contrast to least squares, the PLS coefficients are nonlinear estimators. 
Standardized coefficients indicate the importance of each predictor in the model and correspond to the standardized :math:`x`- and :math:`z`-variables. 
In PLS, the coefficient matrix of shape :math:`(p,)` is calculated from the weights and loadings.

The formula for standardized coefficients is:

.. math::
    \beta^{std} = W\left(P^{T}W\right)^{-1}Q^{T}

To calculate the nonstandardized coefficients and intercept, use these formulas:

.. math::
    \beta_{j} & = \beta_{j}^{std} \dfrac{\sigma_{Z}}{\sigma_{j}} \\
     \beta_{0} & = \mu_{Z} - \displaystyle \sum_{j} \mu_{j}\beta_{j}

where:

==========  ======================================================
Terms       Description
==========  ======================================================
:math:`W`   the :math:`x`-weight matrix
:math:`P`   the :math:`x`-loading matrix
:math:`Q`   the :math:`Z`-loading matrix
:math:`j`   the features :math:`j`
:math:`p`   the number of features
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

.. [1]  R. Tomassone, M. Danzart, J.J. Daudin, J.P. Masson, « Discrimination et classement », Masson, 1988 ; page 38.

.. [2] NonLinear Iterative Partial Least Squares