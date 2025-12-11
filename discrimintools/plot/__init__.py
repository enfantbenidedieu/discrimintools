# -*- coding: utf-8 -*-
from __future__ import annotations

from .fviz_dist import fviz_dist
from .fviz_candisc import fviz_candisc_ind, fviz_candisc_var, fviz_candisc_biplot, fviz_candisc
from .fviz_plsr import fviz_plsr_ind, fviz_plsr_var, fviz_plsr
from .fviz_dica import fviz_dica_ind, fviz_dica_var, fviz_dica_quali_var, fviz_dica_biplot, fviz_dica
from .fviz import add_scatter, set_axis, overlap_coord, fviz_circle

__all__ = [
    "fviz_dist",

    "fviz_candisc_ind",
    "fviz_candisc_var",
    "fviz_candisc_biplot",
    "fviz_candisc",

    "fviz_dica_ind",
    "fviz_dica_var",
    "fviz_dica_quali_var",
    "fviz_dica_biplot",
    "fviz_dica",

    "fviz_plsr_ind",
    "fviz_plsr_var",
    "fviz_plsr",
    
    "add_scatter",
    "set_axis",
    "overlap_coord",
    "fviz_circle"
]