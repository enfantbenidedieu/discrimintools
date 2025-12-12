# -*- coding: utf-8 -*-
from itertools import chain, combinations

def powerset(ls):
    """
    Power Set Generation

    Parameters
    ----------
    ls : list, tuple
        Set of elements.

    Returns
    -------
    powerset: list
        The set of all subsets of the set, without the empty set and including the set itself.
    """
    s = list(ls)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]