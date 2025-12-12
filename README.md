<p align="center">
    <img src="./docs/_static/discrimintools.svg" height=300></img>
</p>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/discrimintools.svg?color=dark-green)](https://pypi.org/project/discrimintools/)
[![Python versions](https://img.shields.io/pypi/pyversions/discrimintools.svg)](https://pypi.org/project/discrimintools/)
[![GitHub](https://shields.io/badge/license-MIT-informational)](https://github.com/enfantbenidedieu/discrimintools/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/discrimintools)](https://pepy.tech/project/discrimintools)
[![Downloads](https://static.pepy.tech/badge/discrimintools/month)](https://pepy.tech/project/discrimintools)
[![Downloads](https://static.pepy.tech/badge/discrimintools/week)](https://pepy.tech/project/discrimintools)

</div>

# discrimintools : Python library for Discriminant Analysis (DA)

discrimintools is an open source [Python](https://www.python.org/) package dedicated to Discriminant Analysis (DA) distributed under the [MIT License](https://github.com/enfantbenidedieu/scientisttools/blob/master/LICENSE.txt).

# Contents

**1. [Overview](#overview)**

**2. [Installation](#installation)**

* [2.1 Global environmen](#genv)
* [2.2 Virtual environment](#venv)
* [2.3 Version](#version)
* [2.4 Dependencies](#dependencies)

**3. [Example](#example)**

**4. [Documentation](#doc)**

**5. [About us](#about_us)**

* [5.1 Authors](#authors)
* [5.2 Feedbacks](#authors)
* [5.3 Citing discrimintools](#citing)

## Overview <a name="overview"></a>

Discriminant analysis is a classification problem, where two or more groups or clusters or populations are known _a priori_ and one or more new observations are classified into one of the known populations based on the measured characteristics.

discrimintools provides functions for:

1. **Discriminant Analysis (DA)**:
    * Canonical Discriminant Analysis - [CANDISC](https://support.sas.com/documentation/onlinedoc/stat/131/candisc.pdf)
    * Discriminant Correspondence Analysis - [DiCA](https://personal.utdallas.edu/~herve/Abdi-DCA2007-pretty.pdf)
    * Discriminant Analysis (linear & quadractic) - [DISCRIM](https://support.sas.com/documentation/onlinedoc/stat/132/discrim.pdf)
    * Stepwise Discriminant Analysis (backward & forward) - [STEPDISC](https://support.sas.com/documentation/onlinedoc/stat/131/stepdisc.pdf)

2. **Factor Analysis (FA)**:
    * General Factor Analysis (PCA, MCA & FAMD)  - [GFA](https://pypi.org/project/scientisttools/)
    * Mixed Principal Component Analysis - [MPCA](https://www.researchgate.net/publication/5087866_Analyse_en_composantes_principales_mixte)

3. **Regularized Discriminant Analysis (RDA)**:
    * Partial Least Squares for Classification - [CPLS](https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/en_Tanagra_PLS_DA.pdf)
    * General Factor Analysis Linear Discriminant Analysis ([PCADA](https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Deploying_Predictive_Models_with_R.pdf), [DISQUAL](https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Pipeline_Python.pdf[) & [DISMIX](https://tutoriels-data-science.blogspot.com/p/tutoriels-en-francais.html#S6wTBImDN7Q)) - [GFALDA](https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/en_dr_utiliser_axes_factoriels_descripteurs.pdf)
    * Discriminant Analysis on Mixed Predictors - [MDA](https://www.researchgate.net/publication/265751966_Discriminant_Analysis_on_Mixed_Predictors)
    * Partial Least Squares Discriminant Analysis - [PLSDA](https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/en_Tanagra_PLS_DA.pdf)
    * Partial Least Squares Logistic Regression - [PLSLOGIT](https://inria.hal.science/inria-00494857v1/document)
    * Partial Least Squares Linear Discriminant Analysis - [PLSDA](https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/en_Tanagra_PLS_DA.pdf)

## Installation <a name="installation"></a>

### Global environment <a name="genv"></a>

You can directly install discrimintools using pip :

```bash
pip install discrimintools
```

or set a virtual environment.

### Virtual environment <a name="venv"></a>

Install the 64-bit version of Python 3, for instance from the [official website](https://www.python.org/). Now create a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) and install discrimintools.

The virtual environment is optional but strongly recommended, in order to avoid potential conflicts with other packages.

```{bash}
PS C:\> python -m venv discrimintools-env # create virtual env
PS C:\> discrimintools-env\Scripts\activate  # activate
PS C:\> pip install -U discrimintools  # install discrimintools
```

### Version <a name="version"></a>

In order to check your installation, you can use.

```{python}
import discrimintools
print(discrimintools.__version__)
```

Using an isolated environment such as *pip venv* or *conda* makes it possible to install a specific version of discrimintools with pip and conda and its dependencies independently of any previously installed Python packages.

You should always remember to activate the environment of your choice prior to running any Python command whenever you start a new terminal session.

### Dependencies <a name="dependencies"></a>

discrimintools is compatible with python version which supports both dependencies :

| Packages          |  Version |
| :---------------- | :------: |
| statsmodels       |  0.14.6  |
| scikit-learn      |  1.8.0   | 
| openpyxl          |  3.1.5   |
| tabulate          |  0.9.0   |
| plotnine          |  0.15.1  |
| adjustText        |  1.3.0   |

## Example <a name="example"></a>

We performs a linear discriminant analysis with ``alcools`` dataset.

```python
from discrimintools.datasets import load_alcools
from discrimintools import DISCRIM
D = load_alcools() # load training data
y, X = D['TYPE'], D.drop(columns=['TYPE']) # split into X and y
clf = DISCRIM()
clf.fit(X,y)
```

## Documentation <a name="doc"></a>

The official documentation is hosted on [https://discrimintools.readthedocs.io](https://discrimintools.readthedocs.io).

## About Us <a name="about_us"></a>

### Authors <a name="authors"></a>

discrimintools is developed and maintained by [Duvérier DJIFACK ZEBAZE](https://www.linkedin.com/in/duv%C3%A9rier-djifack-z-030097118/), the founder 
of djifacklab (*Djifack Laboratory of Mathematics, Statistics and Economics books and packages production using Python Programming Language*).

The djifacklab laboratory maintains others python librairies such as [scientisttools](https://pypi.org/project/scientisttools/), [scientistmetrics](https://pypi.org/project/scientistmetrics/), [scientistshiny](https://pypi.org/project/scientistshiny/), [scientisttseries](https://pypi.org/project/scientistshiny/) and [ggcorrplot]( https://pypi.org/project/ggcorrplot/).

### Feedbacks <a name="feedbacks"></a>

If you have found discrimintools useful in your work, research, or company, please let us know by writing to email [djifacklab@gmail.com](mailto:djifacklab@gmail.com).

### Citing discrimintools <a name="citing"></a>

If discrimintools has been significant in your research, and you would like to acknowledge the project in your academic publication, we suggest citing it using the following *BibTeX format*:

```
@misc{DJIFACK ZEBAZE_2024, 
    url = {https://github.com/enfantbenidedieu/discrimintools}, 
    title = {discrimintools: a Python library for Discriminant Analysis}
    author = {DJIFACK ZEBAZE, Duvérier}, 
    year = {2024}
}
```

   