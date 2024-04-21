# -*- coding: utf-8 -*-
from __future__ import annotations

# Canonical Discriminant Analysis (CANDISC)
from .candisc import CANDISC
from .get_candisc import get_candisc_ind, get_candisc_var, get_candisc_coef, get_candisc,summaryCANDISC
from .fviz_candisc import fviz_candisc

# Linear Discriminant Analysis (LDA)
from .lda import LDA
from .get_lda import get_lda_ind, get_lda_cov, get_lda_coef, get_lda,summaryLDA

# Discriminant Analysis for qualitatives variables (DISQUAL)
from .disqual import DISQUAL

# Discriminant Analysis of Mixed Data (DISMIX)
from .dismix import DISMIX

# Discriminant Correspondence Analysis (DISCA)
from .disca import DISCA
from .get_disca import get_disca_ind, get_disca_var, get_disca_coef, get_disca_classes, summaryDISCA, get_disca
from .fviz_disca import fviz_disca_ind, fviz_disca_mod

# PCA - DA
from .pcada import PCADA

# Stepwise discriminant analysis
from .stepdisc import STEPDISC

from .version import __version__
__name__ = "discrimintools"
__author__ = 'Duvérier DJIFACK ZEBAZE'
__email__ = 'duverierdjifack@gmail.com'