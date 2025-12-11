# -*- coding: utf-8 -*-

def check_is_valid_axis(
        obj,axis
):
    """
    Performs is_valid_axis validation
    ---------------------------------

    Parameters
    ----------
    obj : class
        A class

    axis : list  
        Dimensions to be plotted
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if n_components is equal or greater than 2
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.call_.n_components < 2:
        raise ValueError("n_components must be equal or greater than 2.")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if axis is an instance of list
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(axis,list):
        raise TypeError("'axis' must be a list")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ((len(axis) !=2) or (axis[0] < 0) or (axis[1] > obj.call_.n_components-1)  or (axis[0] > axis[1])):
        raise ValueError("You must pass a valid 'axis'.")

def check_is_valid_geom(
        geom,choice
):
    """
    Check is_valid_geom validation
    ------------------------------

    Parameters
    ----------
    geom : str, list or tuple
        Geometry to be used for the graph.

    choice : list or tuple
        Possible values.  
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if geom is valid
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(geom,str):
        if geom not in choice:
            raise ValueError("The specified value for the argument 'geom' are not allowed")
    elif isinstance(geom,(list,tuple)):
        intersect = [x for x in geom if x in choice]
        if len(intersect)==0:
            raise ValueError("The specified value(s) for the argument geom are not allowed")
    else:
        pass