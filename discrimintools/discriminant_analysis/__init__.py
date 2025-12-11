# -*- coding: utf-8 -*-
from __future__ import annotations

from ._candisc import CANDISC
from ._discrim import DISCRIM
from ._dica import DiCA
from ._gfa import GFA
from ._gfalda import GFALDA
from ._mpca import MPCA
from ._mda import MDA
from ._cpls import CPLS
from ._plsda import PLSDA
from ._plslda import PLSLDA
from ._plslogit import PLSLOGIT
from ._stepdisc import STEPDISC

__all__ = [
    "CANDISC",
    "DISCRIM",
    "DiCA",
    "GFA",
    "GFALDA",
    "MPCA",
    "MDA",
    "CPLS",
    "PLSDA",
    "PLSLDA",
    "PLSLOGIT",
    "STEPDISC"
]