# -*- coding: utf-8 -*-
from __future__ import annotations

#extract elements
from ._eval_predict import eval_predict
from ._summarycandisc import summaryCANDISC
from ._summarycpls import summaryCPLS
from ._summaryda import summaryDA
from ._summarydica import summaryDiCA
from ._summarydiscrim import summaryDISCRIM
from ._summarygfa import summaryGFA
from ._summarygfalda import summaryGFALDA
from ._summarymda import summaryMDA
from ._summarympca import summaryMPCA
from ._summaryplsda import summaryPLSDA
from ._summaryplslda import summaryPLSLDA
from ._summaryplslogit import summaryPLSLOGIT
from ._summarystepdisc import summarySTEPDISC

__all__ = [
    "eval_predict",
    "summaryCANDISC",
    "summaryCPLS",
    "summaryDA",
    "summaryDiCA",
    "summaryDISCRIM",
    "summaryGFA",
    "summaryGFALDA",
    "summaryMDA",
    "summaryMPCA",
    "summaryPLSDA",
    "summaryPLSLDA",
    "summaryPLSLOGIT",
    "summarySTEPDISC"
]