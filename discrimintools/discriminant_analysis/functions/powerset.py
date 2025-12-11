# -*- coding: utf-8 -*-
from itertools import chain, combinations

def powerset(ls):
    """
    Power Set Generation
    --------------------

    Parameters
    ----------
    `ls`: a list or a tuple containing a set of elements

    Returns
    -------
    a list containing the set of all subsets of the set, without the empty set and including the set itself.

    Authors
    -------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    s = list(ls)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]