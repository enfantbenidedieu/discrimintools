*******
DISCRIM
*******

Overview
--------

Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) are two classifiers, with, as their names suggest, a linear and a quadratic decision surface, respectively.

For a set of observations containing one or more quantitative variables and a classification variable defining
groups of observations, the DISCRIM class develops a discriminant criterion to classify each observation
into one of the groups.

The DISCRIMN assume that the distribution within each group is multivariate normal. The discriminant function, also known as a classification criterion,
is determined by a measure of generalized squared distance (Rao 1973). The classification criterion can be
based on either the individual within-group covariance matrices (yielding a quadratic function) or the pooled
covariance matrix (yielding a linear function); it also takes into account the prior probabilities of the groups.

Mathematical formula of the LDA and QDA
---------------------------------------

Both LDA and QDA cen be derived from simple probabilistic models which model the class conditional distribution of the :math:`\mathbb{P}\left(X|Y=y_{k}\right)` for each class :math:`k`. 
Predictions can be obtained by using Bayes' rule, for each training sample :math:`x \in \mathcal{R}^{p}`.

.. math::
    \mathbb{P}\left(y=k|x\right) = \dfrac{\mathbb{P}\left(x|y=k\right)\mathbb{P}(y=k)}{\mathbb{P}\left(x\right)} =  \dfrac{\mathbb{P}\left(x|y=k\right)\mathbb{P}(y=k)}{\displaystyle \sum_{l}\mathbb{P}\left(x|y=l\right)\mathbb{P}(y=l)}

and we select the class :math:`k` which maximizes this posterior probability.

More specifically, for linear and quadratic discriminant analysis, :math:`\mathbb{P}\left(x|y\right)` is modeled as a multivariate gaussian distribution with density:

.. math::
    \mathbb{P}\left(x|y=k\right) = \dfrac{1}{\left(2\pi\right)^{p/2}\lvert \Sigma_{k}\rvert^{1/2}}\text{exp}\left(-\dfrac{1}{2}\left(x-\mu_{k}\right)^{T} \Sigma_{k}^{-1}\left(x-\mu_{k}\right)\right)

Quadratic Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to the model above, the log of the posterior probability is:

.. math::
    \begin{eqnarray}
    \log \mathbb{P}\left(y=k|x\right) & = & \log \mathbb{P}\left(x|y=k\right)+\log \mathbb{P}\left(y=k\right) - \log \mathbb{P}(x)\\
     & = & - \dfrac{p}{2}\log\left(2\pi\right)-\dfrac{1}{2}\log \lvert \Sigma_{k} \rvert -\dfrac{1}{2}\left(x-\mu_{k}\right)^{T} \Sigma_{k}^{-1}\left(x-\mu_{k}\right) +\log \mathbb{P}\left(y=k\right) - \log \mathbb{P}(x) \\
     & = & -\dfrac{1}{2}\log \lvert \Sigma_{k} \rvert -\dfrac{1}{2}\left(x-\mu_{k}\right)^{T} \Sigma_{k}^{-1}\left(x-\mu_{k}\right) +\log \mathbb{P}\left(y=k\right) + Cst
    \end{eqnarray}

The predicted class is the one that maximises this log-posterior probability.

Linear Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LDA is a special case of QDA, where the Gaussian for each class are assumed to share the same covariance matrix: :math:`\Sigma_{k}=\Sigma` for all :math:`k`. This reduces the log posterior probability to:

.. math::
    \log \mathbb{P}\left(y=k|x\right) = -\dfrac{1}{2}\left(x-\mu_{k}\right)^{T} \Sigma^{-1}\left(x-\mu_{k}\right) +\log \mathbb{P}\left(y=k\right) + Cst

The term :math:`\left(x-\mu_{k}\right)^{T} \Sigma^{-1}\left(x-\mu_{k}\right)` corresponds to the `Mahalanobis <https://en.wikipedia.org/wiki/Mahalanobis_distance>`_ between the sample :math:`x` and the mean :math:`\mu_{k}`. 

.. note::
    The Mahalanobis distance tells how close :math:`x` is from :math:`\mu_{k}`, while also accounting for the variance of each feature. WE can thus interpret LDA as assigning 
    :math:`x` to the class whose mean is the closest in terms of Mahalanobis distance, while also accounting for the class prior probabilities.

The log posterior probability of LDA can also be written [1]_ as:

.. math::
    \log \mathbb{P}\left(y=k|x\right) = \beta_{k0} + \beta_{k}^{T}x

where :math:`\beta_{k} = \Sigma^{-1}\mu_{k}` and :math:`\beta_{k0} = \log \mathbb{P}\left(y=k\right) - \dfrac{1}{2}\mu_{k}^{T}\Sigma^{-1}\mu_{k}`.

.. note::
    From the above formula, it is clear that linear discriminant analysis has a linear decision surface. In this case of quadratic discriminant analysis, the are no assumptions on the covariance matrices :math:`\Sigma_{k}` of the 
    Gaussians, leading to quadratic decision surface. See [2]_ for more details.

Estimating Parameters of Normal Distributions
---------------------------------------------

In any practical use, we need to estmate some of these quantities:

* Priors probabilities: :math:`\widehat{\pi}_{k}`
* Mean-vectors: :math:`\widehat{\mu}_{k}`
* Variance-covariance matrices : :math:`\widehat{\Sigma}_{k}`

Priors
^^^^^^

Estimating :math:`\pi_{k}` is relatively intuitive:

.. math::
    \widehat{\pi}_{k} = \dfrac{n_{k}}{n}

where :math:`n_{k} = lvert \mathcal{C}_{k}\rvert` denotes the size of class :math:`k` and :math:`n` denotes the total number of data points.

Mean vectors
^^^^^^^^^^^^^

For :math:`\widehat{\mu}_{k}`, we can use the centroid of :math:`\mathcal{C}_{k}`; i.e. the average individual of class :math:`k`:

.. math::
    \widehat{\mu}_{k} = g_{k}

Variance-Covariance matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For :math:`\widehat{\Sigma}_{k}`, we can use the within-variance matrix:

.. math::
    \widehat{\Sigma}_{k} = \dfrac{1}{n_{k}-1} X_{k}^{T}X_{k}

where :math:`X_{k}` is the mean-centered data matrix for objects of class :math:`\mathcal{C}_{k}`.

Given all of above estimations, we can estimate the posterior probability.

Quadratic Discriminant Analysis
"""""""""""""""""""""""""""""""

For quadratic discriminant analysis, we have :

.. math::
    \widehat{\delta}_{k}\left(x\right) = \underbrace{-\dfrac{1}{2}x^{T}\widehat{\Sigma}_{k}^{-1}x}_{\text{quadratic}} - \underbrace{x^{T}\widehat{\Sigma}_{k}^{-1}\widehat{\mu}_{k}}_{\text{linear}} + \underbrace{\log \left(\widehat{\pi}_{k}\right) - \dfrac{1}{2}\log \lvert \widehat{\Sigma}_{k} \rvert - \dfrac{1}{2}\widehat{\mu}_{k}^{T}\widehat{\Sigma}_{k}^{-1}\widehat{\mu}_{k}}_{\text{constant}}

Having a quadratic discriminant function causes the decision boundaries in quadratic discriminant analysis to be quadratic surfaces.

Linear Discriminant Analysis
""""""""""""""""""""""""""""

For linear discriminant analysis, since all covariances matrices are the same, the pooled within-class matrix is defined as :

.. math::
    \widehat{\Sigma} = \dfrac{1}{n-K} \displaystyle \sum_{k} \left(n_{k}-1\right)\widehat{\Sigma}_{k} 

Our expression for :math:`\widehat{\delta}_{k}\left(x\right)`, becomes (after ignoring termes that do not depend on :math:`k`):

.. math::
    \widehat{\delta}_{k}\left(x\right) =  - \underbrace{x^{T}\widehat{\Sigma}^{-1}\widehat{\mu}_{k}}_{\text{linear}} + \underbrace{\log \left(\widehat{\pi}_{k}\right) - \dfrac{1}{2}\widehat{\mu}_{k}^{T}\widehat{\Sigma}^{-1}\widehat{\mu}_{k}}_{\text{constant}}

For more details about linear and quadratic discriminant analysis, see SAS `DISCRIM Procedure <https://support.sas.com/documentation/onlinedoc/stat/132/discrim.pdf>`_.

.. rubric:: References

.. [1]  R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification (Second Edition), section 2.6.2.

.. [2] “The Elements of Statistical Learning”, Hastie T., Tibshirani R., Friedman J., Section 4.3, p.106-119, 2008.