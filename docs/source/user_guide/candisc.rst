*******
CANDISC
*******

Overview
--------

Canonical discriminant analysis is a dimension-reduction technique related to principal component analysis and canonical correlation. 
The methodology that is used in deriving the canonical coefficients parallels that of a one-way multivariate analysis of variance (MANOVA). 
MANOVA tests for equality of the mean vector across class levels. 
Canonical discriminant analysis finds linear combinations of the quantitative variables that provide maximal separation between classes or groups.

CANDISC is a somewhat hybrid learning nature with two setps:

1. **Semi-supervised**: On one hand, CANDISC has an unsupervised or descriptive aspect that aims to tackle the following question. How to find a representation of the objects which provides the best separation between classes?
2. **Supervised**: On the other hand, CANDISC also has a decisively supervised aspect that tackles the question: How to find the rules for assigning a class to a given object?

Semi-supervised Aspect
----------------------

Mathematical formula of the CANDISC classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Semi-supervised aspect, the gold is to find a low dimensional representation of the objects which provides the best separation between classes. 
We can look for an axis :math:`\Delta_{a}`, spanned by some columns vector :math:`a=(a_{1},a_{2},\dots,a_{p})`, such that the linear combinations [#f1]_:

.. math::
    z = a_{1}(x_{1} - \overline{x}_{1})+a_{2}(x_{2} - \overline{x}_{2})+\cdots+a_{p}(x_{p} - \overline{x}_{p})

that separates all :math:`K` groups in an adequate way.

Algebraically, the idea is to look for a linear combination of the predictors :

.. math::
     Z= (X - \mathbb{1}g^{T})a

that *ideally* could achieve the following two goals [#f2]_:

* Minimize (pooled) within-class dispersion (wss): :math:`\underbrace{\min}_{a}\left\{a^{T}W_{b}a\right\}`
* Maximize between-class dispersion (bss): :math:`\underbrace{\max}_{a}\left\{a^{T}B_{b}a\right\}`

On one hand, it would be nice to have :math:`a`, such that the between-class dispersion is maximized. This corresponds to a situation in which the class centroids are well separated.
On the other hand, it would also make sense to have :math:`a`, such that the within-class dispersion is minimized. This implies having classes in which, on average, the “inner” variation is small (i.e. concentrated local dispersion).

Can we find such a mythical vector :math:`a`?

Looking for a Compromise Criterion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far we have an impossible simultaneity involving a minimization criterion, as well as a maximization criterion:

.. math::
    \begin{eqnarray}
    \underbrace{\min}_{a}\left\{a^{T}W_{b}a\right\} & \Longrightarrow & W_{b}a = \lambda a \\
    & \text{and} & \\
   \underbrace{\max}_{a}\left\{a^{T}B_{b}a\right\} & \Longrightarrow & B_{b}a = \rho a
   \end{eqnarray}

What can we do to look for a compromise. Using the Huygens theoreom, the variance can be deomposed as:

.. math::
    V_{b} = W_{b} + B_{b}

Doing some algebra, it can be shown that the quadratic form :math:`a^{T}V_{b}a` can be decomposed as:

.. math::
    a^{T}V_{b}a = a^{T}W_{b}a + a^{T}B_{b}a

Again, we are pursuing a dual goal that is, in general, hard to accomplish:

.. math::
    a^{T}V_{b}a = \underbrace{a^{T}W_{b}a}_{\text{minimize}} + \underbrace{a^{T}B_{b}a}_{\text{maximize}}

We have two options for the compromise:

.. math::
    \underbrace{\max}_{a}\left\{\dfrac{a^{T}B_{b}a}{a^{T}V_{b}a}\right\} \quad \text{or} \quad \underbrace{\max}_{a}\left\{\dfrac{a^{T}B_{b}a}{a^{T}W_{b}a}\right\}

which are actually associated to the following ratios:

.. math::
    \eta^{2} = \dfrac{a^{T}B_{b}a}{a^{T}V_{b}a} \quad \text{and} \quad F = \dfrac{a^{T}B_{b}a}{a^{T}W_{b}a}

where :math:`\eta^{2}` is the correlation ratio and :math:`F` the :math:`F`-ratio.

Correlation Ratio
"""""""""""""""""

If we decide to work with the first criterion, we look for :math:`a` such that:

.. math::
    \underbrace{\max}_{a}\left\{\dfrac{a^{T}B_{b}a}{a^{T}V_{b}a}\right\} 

This criterion is scale invariant, meaning that we use any scale variation of :math:`a`: i.e. :math:`\alpha a`. For convenience, we can impose a normalizing restriction: :math:`a^{T}V_{b}a=1`. Consequently.

.. math::
    \underbrace{\max}_{a}\left\{\dfrac{a^{T}B_{b}a}{a^{T}V_{b}a}\right\}  \quad \Longleftrightarrow \quad \underbrace{\max}_{a}\left\{a^{T}B_{b}a\right\}\quad \text{s.t.}\quad a^{T}V_{b}a=1

Using the method of Lagrangien multiplier:

.. math::
    \mathcal{l}(a,\lambda) = a^{T}B_{b}a - \lambda(a^{T}V_{b}a - 1)

Deriving w.r.t :math:`a` and equating to zero:

.. math::
    \dfrac{\partial \mathcal{l}}{a} = 2B_{b}a - 2\lambda V_{b}a= 0

The optimal vector :math:`a` is such that:

.. math::
    B_{b}a = \lambda V_{b}a

If the matrix :math:`V_{b}` is inversible, which it is in general, then :

.. math::
    V_{b}^{-1}B_{b}a = \lambda a

that is, the optimal vector :math:`a` is eigenvector of :math:`V_{b}^{-1}B_{b}`. Keep in mind that, in general, :math:`V_{b}^{-1}B_{b}` is not symmetric.

:math:`F`-ratio Criterion
"""""""""""""""""""""""""

Now, if we decide to work with the criterion associated to the `F` ratio, then the criterion to be maximized is:

.. math::
    \underbrace{\max}_{a}\left\{\dfrac{a^{T}B_{b}a}{a^{T}W_{b}a}\right\}  \quad \Longleftrightarrow \quad \underbrace{\max}_{a}\left\{a^{T}B_{b}a\right\} \quad \text{s.t.}\quad a^{T}W_{b}a=1

Applying the same Lagrangien procedure, with a multiplier :math:`\rho`, we have that :math:`a` is such a vector that:

.. math::
    B_{b}a = \rho W_{b}a

and if :math:`W_{b}` is inversible, which it is in most cases, then it can be shown that :math:`a` is also eigenvector of :math:`W_{b}^{-1}B_{b}`, associated to eigenvalue :math:`\rho`:

.. math::
    W_{b}^{-1}B_{b}a = \rho a

Relationshop between eigenvalues
""""""""""""""""""""""""""""""""

The relationship between the eigenvalues :math:`\lambda` and :math:`\rho` is :

.. math::
    \rho = \dfrac{\lambda}{1 - \lambda}

.. note::
    1. :math:`\lambda = \eta^{2}` correspond to the correlation ratio and is range between :math:`0` and :math:`1` (:math:`0 \leq \lambda \leq 1`).
    2. :math:`\sqrt{\lambda} = \eta` correspond to the canonical correlation.
    3. :math:`\lambda` are not additive from one factor to another.
    4. Eiganvalues :math:`\rho` are added together from one factor to another.

Raw Canonical coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^

Coefficients :math:`a_{h} (h=1,\dots,H)` are obtained using the following formula:

.. math::
    a_{h} = \left(V_{b}^{-1}C\right)b_{h}\sqrt{\dfrac{n-K}{n\times \lambda_{h} \times \left( 1 - \lambda_{h}\right)}}

where :math:`b_{h}` is the eigenvector of :math:`C^{T}V_{b}^{-1}C` and :math:`C`, a matrix of shape :math:`(p,K)` such as :math:`V_{b}=CC^{T}` and 

.. math::
    c_{pk} = \sqrt{\dfrac{n_{k}}{n}}\left(\overline{x}_{kj} - \overline{x}_{j}\right)

The intercept, :math:`a_{h0}`, correspond to:

.. math::
    a_{h0} = - \displaystyle \sum_{j=1}^{j=p}a_{hj}\overline{x}_{j}


Supervised Aspect
-----------------
The supervised learning aspect of CANDISC has to do with the question: how do we use it for classification purposes? This involves establishing a decision rule that let us predict the class of an object. CANDISC proposes a geometric rule of classification. 

Distance behind CANDISC
^^^^^^^^^^^^^^^^^^^^^^^

The squared Euclidean distance between two vectors :math:`x_{i}` and :math:`x_{i^{'}}` is defined as:

.. math::
    d_{E}^{2}\left(i,i^{'}\right) = \left(x_{i}-x_{i^{'}}\right)^{T}\left(x_{i}-x_{i^{'}}\right)

The squared Euclidean distance between the vector :math:`x_{i}` and the coordinates of the centroids :math:`g_{k}` is defined as:

.. math::
    d_{E}^{2}\left(i,\mathcal{C}_{k}\right) = \left(x_{i} - g_{k}\right)^{T}\left(x_{i} - g_{k}\right)

The generalized squared distance between the vector :math:`x_{i}` and the coordinates of the centroids :math:`g_{k}` is defined as:

.. math::
    d_{M}^{2}\left(i,\mathcal{C}_{k}\right) = d_{E}^{2}\left(i,\mathcal{C}_{k}\right) - 2 \times \log \left(\widehat{\pi}_{k}\right)

where :math:`\widehat{\pi}_{k}` is the prior probability of class :math:`\mathcal{C}_{k}`.

Predictive idea
^^^^^^^^^^^^^^^

The classification rule used in CANDISC consists of assigning each individual :math:`x_{i}` to the class :math:`\mathcal{C}_{k}` for which the distance to the centroid is minimal. For that, we use a variant of softmax transformation 
for estimated probabilities:

.. math::
    \mathbb{P}\left(Y=y_{k}/X = x_{i}\right) = \dfrac{e^{-0.5\times d_{M}^{2}\left(i,\mathcal{C}_{k}\right)}}{\displaystyle \sum_{c=1}^{c=K}e^{-0.5\times d_{M}^{2}\left(i,\mathcal{C}_{c}\right)}} 


Classification Functions Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the generalized square distance, it is possible to deduct the classification function coefficients.

.. math::

    \begin{eqnarray}
      -\dfrac{1}{2}d_{M}^{2}\left(i,\mathcal{C}_{k}\right) & = & - \dfrac{1}{2}\displaystyle \sum_{h=1}^{h=H}\left(z_{h}(i) - \overline{z}_{kh}\right)^{2}+\log \left(\widehat{\pi}_{k}\right)\\
      & = & -\dfrac{1}{2}\displaystyle \sum_{h=1}^{h=H}z_{h}(i)^{2} -\dfrac{1}{2}\displaystyle \sum_{h=1}^{h=H}\overline{z}_{kh}^{2} + \displaystyle \sum_{h=1}^{h=H}z_{h}(i)\overline{z}_{kh} + \log \left(\widehat{\pi}_{k}\right)
   \end{eqnarray}

Since the canonical discriminant function is a linear function of original variables :

.. math::
    z_{h}(x) = a_{h0}+a_{h1}x_{1}+a_{h2}x_{2}+\cdots+a_{hp}x_{p}

we can deduct the linear expression of the classification function for the CANDISC:

.. math::
    S\left(y_{k},i\right)  =  \beta_{k0} + \beta_{k1}x_{1} + \beta_{k2}x_{2} + \cdots + \beta_{kp}x_{p} 

where :math:`\beta_{k0} = \log \left(\widehat{\pi}_{k}\right) + \displaystyle \sum_{h}^{h=H}a_{h0}\overline{z}_{kh} -\dfrac{1}{2}\displaystyle \sum_{h=1}^{h=H}\overline{z}_{kh}^{2}` and 
:math:`\beta_{kj} = \displaystyle \sum_{h=1}^{h=H}a_{hj}\overline{z}_{kh}`.

For more details about canonical discriminant analysis, see SAS `CANDISC Procedure`_.

.. _CANDISC Procedure: https://support.sas.com/documentation/onlinedoc/stat/131/candisc.pdf

.. rubric:: Footnotes

.. [#f1] Intercept is not need since variables are centered.
.. [#f2] The index :math:`b` mean that it is weighted by :math:`1/n` and not its degree of freedom.

.. rubric:: References