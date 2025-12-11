# -*- coding: utf-8 -*-
from pandas import concat
from plotnine import ggplot, aes, geom_point, geom_text, scale_color_manual

#interns functions
from .utils import check_is_valid_axis, check_is_valid_geom
from .fviz import add_scatter, set_axis, list_colors
from .fviz_dist import fviz_dist

def fviz_dica_ind(
        obj,
        axis = [0,1],
        geom_ind = ("point","text"),
        repel = False,
        point_args_ind = dict(shape = "o", size = 1.5),
        text_args_ind = dict(size = 8),
        add_group = True,
        geom_group = ("point","text"),
        point_args_group = dict(shape = "^", size = 3),
        text_args_group = dict(size = 11.5),
        palette = None,
        x_lim = None,
        y_lim = None,
        x_label = None,
        y_label = None,
        title = None,
        add_hline = True,
        add_vline = True,
        add_grid = True,
        ggtheme = None
):
    """
    Visualize Discriminant Correspondence Analysis (DiCA) - Graph of individuals

    Discriminant correspondence analysis (DiCA) is a canonical discriminant analysis on qualitative predictors. :class:`~discrimintools.plot.fviz_dica_ind` provides plotnine based elegant visualization of DiCA outputs for individuals.

    Parameters
    ----------
    obj : class
        An instance of class :class:`~discrimintools.discriminant_analysis.DiCA`
    
    axis : list, defaul = [0,1]
        Dimensions to be plotted

    geom_ind : str, list or tuple, default = ('point','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["point","text"]. 

        - 'point' to show only points,
        - 'text' to show only labels,
        - ('point','text') to show both types.

    repel : bool, default=False 
        To avoid overplotting text labels.

    point_args_ind : dict, default = dict(shape = "o", size = 1.5)
        Keywords arguments for `geom_point <https://plotnine.org/reference/geom_point.html>`_.
    
    text_args_ind : dict, default = dict(size = 8)
        Keywords arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.

    add_group : bool, default = True
        To show group coordinates.

    geom_group : str, list or tuple, default = ('point','text')
        See ``geom_ind``.

    point_args_group : dict, default = dict(shape = "^", size = 3)
        See ``point_args_ind``.
        
    text_args_group : dict, default = dict(size = 11.5)
        See ``text_args_ind``.
    
    palette : None or list, default = None
        Color palette to be used for coloring by groups.
    
    x_lim : None, list or tuple, default = None
        The range of the plotted ``x`` values.
    
    y_lim : None, list or tuple, default = None
        The range of the plotted ``y`` values.
    
    x_label : None or str, default = None
        The label text of ``x``.
    
    y_label : None or str, default = None
        The label text of ``y``.
    
    title : None or str, default = None
        The title of the graph you draw.
    
    add_hline : bool, default = True
        To add a horizontal line.
    
    add_vline : bool, default = True
        To add a vertical line.
    
    add_grid : bool, default = True
        To add grid customization.
        
    ggtheme : function, default = None
        Plotnine `theme <https://plotnine.org/guide/themes-premade.html>`_ name.
    
    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.plot.fviz_dica`
        Visualize Discriminant Correspondence Analysis (DiCA).
    :class:`~discrimintools.plot.fviz_dica_biplot`
        Visualize Discriminant Correspondence Analysis (DiCA) - Biplot of individuals and variables.
    :class:`~discrimintools.plot.fviz_dica_quali_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of qualitative variables.
    :class:`~discrimintools.plot.fviz_dica_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - graph of variables/categories.
    :class:`~discrimintools.plot.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA, fviz_dica_ind
    >>> D = load_divay() # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    >>> p = fviz_dica_ind(clf) # graph of individuals
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_ind.png
        :scale: 90%
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class DiCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "dica":
        raise TypeError("'obj' must be an instance of class DiCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid iaxis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom_ind,('point','text'))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set text arguments - add overlap arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and "text" in geom_ind:
        text_args_ind = dict(**text_args_ind,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index and palette
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index
    index = obj.call_.classes
    #set palette
    if palette is None:
        palette = list_colors[:len(index)]
    elif not isinstance(palette,(list,tuple)):
        raise TypeError("'palette' must be a list or a tuple of colors")
    elif len(palette) != len(index):
        raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    #create coordinate
    coord = concat((obj.ind_.coord,obj.call_.y),axis=1)
    #initialize
    p = ggplot(data=coord,mapping=aes(x = f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",label=coord.index))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "point" in geom_ind:
        p = p + geom_point(aes(color=obj.call_.target),**point_args_ind)
    if "text" in geom_ind:
        p = p + geom_text(aes(color=obj.call_.target),**text_args_ind)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add classes coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if add_group:
        #check if valid geom
        check_is_valid_geom(geom_group,('point','text'))

        #set text arguments - add overlap arguments
        if repel and "text" in geom_group:
            text_args_group = dict(**text_args_group,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))

        #classes coordinates
        class_coord = obj.classes_.coord
        class_coord[f"{obj.call_.target}"] = list(class_coord.index)
        #add points
        if "point" in geom_group:
            p = p + geom_point(data=class_coord,mapping=aes(x = f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",color=obj.call_.target,label=class_coord.index),**point_args_group)
        #add texts
        if "text" in geom_group:
            p = p + geom_text(data=class_coord,mapping=aes(x = f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",color=obj.call_.target,label=class_coord.index),**text_args_group)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add color scale
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = p + scale_color_manual(values=palette,labels=index)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set x label
    if x_label is None:
        x_label = "Can{} ({}%)".format(axis[0]+1,round(obj.eig_.iloc[axis[0],2],1))
    #set y label
    if y_label is None:
        y_label = "Can{} ({}%)".format(axis[1]+1,round(obj.eig_.iloc[axis[1],2],1))
    #set title
    if title is None:
        title = "Graph of individuals - {}".format(obj.__class__.__name__)
    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_dica_var(
        obj,
        axis = [0,1],
        geom_var = ("point","text"),
        repel = False,
        col_var = "blue",
        point_args_var = dict(shape = "o", size = 1.5),
        text_args_var = dict(size = 8),
        add_group = True,
        geom_group = ("point","text"),
        point_args_group = dict(shape = "^", size = 3),
        text_args_group = dict(size = 11.5),
        palette = None,
        x_lim = None,
        y_lim = None,
        x_label = None,
        y_label = None,
        title = None,
        add_hline = True,
        add_vline = True,
        add_grid = True,
        ggtheme = None
):
    """
    Visualize Discriminant Correspondence Analysis (DiCA) - Graph of variables/categories

    Discriminant correspondence analysis (DiCA) is a canonical discriminant analysis on qualitative predictors. :class:`~discrimintools.plot.fviz_dica_var` provides plotnine based elegant visualization of DiCA outputs for variables/categories.

    Parameters
    ----------
    obj : class
        An instance of class :class:`~discrimintools.discriminant_analysis.DiCA`.
    
    axis : list, defaul = [0,1]
        Dimensions to be plotted.

    geom_var : str, list or tuple, default = ('point','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["point","text"]. 

        - 'point' to show only points,
        - 'text' to show only labels,
        - ('point','text') to show both types.

    repel : bool, default = False 
        To avoid overplotting text labels.

    col_var : str, default = 'blue'
        Color for the variables points and texts.

    point_args_var : dict, default = dict(shape = "o", size = 1.5)
        Keywords arguments for `geom_point <https://plotnine.org/reference/geom_point.html>`_.
    
    text_args_var : dict, default = dict(size = 8)
        KeywordS arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.

    add_group : bool, default = True
        To show group coordinates.

    geom_group : str, list or tuple, default = ('point','text')
        See ``geom_ind``.

    point_args_group : dict, default = dict(shape = "^", size = 3)
        See ``point_args_ind``.
        
    text_args_group : dict, default = dict(size = 11.5)
        See ``text_args_ind``.
    
    palette : None or list, default = None
        Color palette to be used for coloring by groups.
    
    x_lim : None, list or tuple, default = None
        The range of the plotted ``x`` values.
    
    y_lim : None, list or tuple, default = None
        The range of the plotted ``y`` values.
    
    x_label : None or str, default = None
        The label text of ``x``.
    
    y_label : None or str, default = None
        The label text of ``y``.
    
    title : None or str, default = None
        The title of the graph you draw.
    
    add_hline : bool, default = True
        To add a horizontal line.
    
    add_vline : bool, default = True
        To add a vertical line.
    
    add_grid : bool, default = True
        To add grid customization.
        
    ggtheme : function, default = None
        Plotnine `theme <https://plotnine.org/guide/themes-premade.html>`_ name.

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.plot.fviz_dica`
        Visualize Discriminant Correspondence Analysis (DiCA).
    :class:`~discrimintools.plot.fviz_dica_biplot`
        Visualize Discriminant Correspondence Analysis (DiCA) - Biplot of individuals and variables.
    :class:`~discrimintools.plot.fviz_dica_ind`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of individuals.
    :class:`~discrimintools.plot.fviz_dica_quali_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of qualitative variables.
    :class:`~discrimintools.plot.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA, fviz_dica_var
    >>> D = load_divay() # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    >>> p = fviz_dica_var(clf) # graph of variables
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_var.png
        :scale: 90%
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class DiCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "dica":
        raise TypeError("'obj' must be an instance of class DiCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom_var,('point','text'))
    
    #initialize
    p = ggplot()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables/categories coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(p=p,data=obj.var_.coord,axis=axis,geom=geom_var,repel=repel,color=col_var,points_args=point_args_var,text_args=text_args_var)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add classes coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if add_group:
        #check if valid geom
        check_is_valid_geom(geom_group,('point','text'))

        #set text arguments - add overlap arguments
        if repel and "text" in geom_group:
            text_args_group = dict(**text_args_group,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))

        #set index
        index = obj.call_.classes
        #set palette
        if palette is None:
            palette = list_colors[:len(index)]
        elif not isinstance(palette,(list,tuple)):
            raise TypeError("'palette' must be a list or a tuple of colors")
        elif len(palette) != len(index):
            raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
        
        #classes coordinates
        class_coord = obj.classes_.coord
        class_coord[f"{obj.call_.target}"] = list(class_coord.index)
        #add points
        if "point" in geom_group:
            p = p + geom_point(data=class_coord,mapping=aes(x = f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",color=obj.call_.target,label=class_coord.index),**point_args_group)
        #add texts
        if "text" in geom_group:
            p = p + geom_text(data=class_coord,mapping=aes(x = f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",color=obj.call_.target,label=class_coord.index),**text_args_group)
        #add color scale
        p = p + scale_color_manual(values=palette,labels=index)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set x label
    if x_label is None:
        x_label = "Can{} ({}%)".format(axis[0]+1,round(obj.eig_.iloc[axis[0],2],1))
    #set y label
    if y_label is None:
        y_label = "Can{} ({}%)".format(axis[1]+1,round(obj.eig_.iloc[axis[1],2],1))
    #set title
    if title is None:
        title = "Graph of variables/categories - {}".format(obj.__class__.__name__)
    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_dica_quali_var(
        obj,
        axis = [0,1],
        geom_var = ("point","text"),
        repel = False,
        col_var = "blue",
        point_args_var = dict(shape = "o", size = 1.5),
        text_args_var = dict(size = 8),
        x_lim = (0,1),
        y_lim = (0,1),
        x_label = None,
        y_label = None,
        title = None,
        add_hline = True,
        add_vline = True,
        add_grid = True,
        ggtheme = None
):
    """
    Visualize Discriminant Correspondence Analysis (DiCA) - Graph of qualitative variables

    Discriminant correspondence analysis (DiCA) is a canonical discriminant analysis on qualitative predictors. :class:`~discrimintools.plot.fviz_dica_quali_var` provides plotnine based elegant visualization of DiCA outputs for qualitative variables.

    Parameters
    ----------
    obj : class
        An instance of class :class:`~discrimintools.discriminant_analysis.DiCA`.
    
    axis : list, defaul = [0,1]
        Dimensions to be plotted.

    geom_var : str, list or tuple, default = ('point','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["point","text"]. 

        - 'point' to show only points,
        - 'text' to show only labels,
        - ('point','text') to show both types.

    repel : bool, default = False 
        To avoid overplotting text labels.

    col_var : str, default = 'blue'
        Color for the qualitative variables points and texts.

    point_args_var : dict, default = dict(shape = "o", size = 1.5)
        Keywords arguments for `geom_point <https://plotnine.org/reference/geom_point.html>`_.
    
    text_args_var : dict, default = dict(size = 8)
        Keywords arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.

    x_lim : None, list or tuple, default = (0,1)
        The range of the plotted ``x`` values.
    
    y_lim : None, list or tuple, default = (0,1)
        The range of the plotted ``y`` values.
    
    x_label : None or str, default = None
        The label text of ``x``.
    
    y_label : None or str, default = None
        The label text of ``y``.
    
    title : None or str, default = None
        The title of the graph you draw.
    
    add_hline : bool, default = True
        To add a horizontal line.
    
    add_vline : bool, default = True
        To add a vertical line.
    
    add_grid : bool, default = True
        To add grid customization.
        
    ggtheme : function, default=None
        Plotnine `theme <https://plotnine.org/guide/themes-premade.html>`_ name.

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.plot.fviz_dica`
        Visualize Discriminant Correspondence Analysis (DiCA)
    :class:`~discrimintools.plot.fviz_dica_biplot`
        Visualize Discriminant Correspondence Analysis (DiCA) - Biplot of individuals and variables
    :class:`~discrimintools.plot.fviz_dica_ind`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of individuals
    :class:`~discrimintools.plot.fviz_dica_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of variables/categories
    :class:`~discrimintools.plot.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA, fviz_dica_quali_var
    >>> D = load_divay("train") # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    >>> p = fviz_dica_quali_var(clf) # graph of qualitative variables
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_quali_var.png
        :scale: 90%
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class DiCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "dica":
        raise TypeError("'obj' must be an instance of class DiCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom_var,('point','text'))
    
    #initialize
    p = ggplot()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables/categories coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(p=p,data=obj.var_.eta2,axis=axis,geom=geom_var,repel=repel,color=col_var,points_args=point_args_var,text_args=text_args_var)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set x label
    if x_label is None:
        x_label = "Can{} ({}%)".format(axis[0]+1,round(obj.eig_.iloc[axis[0],2],1))
    #set y label
    if y_label is None:
        y_label = "Can{} ({}%)".format(axis[1]+1,round(obj.eig_.iloc[axis[1],2],1))
    #set title
    if title is None:
        title = "Graph of qualitative variables - {}".format(obj.__class__.__name__)
    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_dica_biplot(
        obj,
        axis = [0,1],
        geom_ind = ("point","text"),
        repel = False,
        point_args_ind = dict(shape = "o", size = 1.5),
        text_args_ind = dict(size = 8),
        geom_var = ("point","text"),
        col_var = "blue",
        point_args_var = dict(shape = "o", size = 1.5),
        text_args_var = dict(size = 8),
        add_group = True,
        geom_group = ("point","text"),
        point_args_group = dict(shape = "^", size = 3),
        text_args_group = dict(size = 11.5),
        palette = None,
        x_lim = None,
        y_lim = None,
        x_label = None,
        y_label = None,
        title = None,
        add_hline = True,
        add_vline = True,
        add_grid = True,
        ggtheme = None
        
):
    """
    Visualize Discriminant Correspondence Analysis (DiCA) - Biplot of individuals and variables

    Discriminant correspondence analysis (DiCA) is a canonical discriminant analysis on qualitative predictors. :class:`~discrimintools.plot.fviz_dica_biplot` provides plotnine based elegant visualization of DiCA outputs for individuals and variables.

    Parameters
    ----------
    obj : class
        An object of class :class:`~discrimintools.discriminant_analysis.DiCA`.

    **kwargs : 
        further arguments passed to or from other methods. See :class:`~discrimintools.plot.fviz_dica_ind`, :class:`~discrimintools.plot.fviz_dica_var`.

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.plot.fviz_dica`
        Visualize Discriminant Correspondence Analysis (DiCA)
    :class:`~discrimintools.plot.fviz_dica_ind`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of individuals
    :class:`~discrimintools.plot.fviz_dica_quali_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of qualitative variables
    :class:`~discrimintools.plot.fviz_dica_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - graph of variables/categories
    :class:`~discrimintools.plot.fviz_dist`
        Visualize distance between barycenter.    

    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA, fviz_dica_biplot
    >>> D = load_divay() # load training data
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()
    >>> p = fviz_dica_biplot(clf) # biplot of individuals and variables
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_biplot.png
        :scale: 90%
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class DiCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "dica":
        raise TypeError("'obj' must be an instance of class DiCA")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid iaxis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom_ind,('point','text'))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set text arguments - add overlap arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and "text" in geom_ind:
        text_args_ind = dict(**text_args_ind,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index and palette
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index
    index = obj.call_.classes
    #set palette
    if palette is None:
        palette = list_colors[:len(index)]
    elif not isinstance(palette,(list,tuple)):
        raise TypeError("'palette' must be a list or a tuple of colors")
    elif len(palette) != len(index):
        raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")
    
    #set x_text and y_text
    x_text, y_text = "Can{}".format(axis[0]+1), "Can{}".format(axis[1]+1)

    #create coordinate
    coord = concat((obj.ind_.coord,obj.call_.y),axis=1)
    #initialize
    p = ggplot(data=coord,mapping=aes(x=x_text,y=y_text,label=coord.index))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add individuals coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "point" in geom_ind:
        p = p + geom_point(aes(color=obj.call_.target),**point_args_ind)
    if "text" in geom_ind:
        p = p + geom_text(aes(color=obj.call_.target),**text_args_ind)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = add_scatter(p=p,data=obj.var_.coord,axis=axis,geom=geom_var,repel=repel,color=col_var,points_args=point_args_var,text_args=text_args_var)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add classes coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if add_group:
        #check if valid geom
        check_is_valid_geom(geom_group,('point','text'))
        #set text arguments - add overlap arguments
        if repel and "text" in geom_group:
            text_args_group = dict(**text_args_group,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))

        #classes coordinates
        class_coord = obj.classes_.coord
        class_coord[f"{obj.call_.target}"] = list(class_coord.index)
        if "point" in geom_group:
            p = p + geom_point(data=class_coord,mapping=aes(x=x_text,y=y_text,color=obj.call_.target,label=class_coord.index),**point_args_group)
        if "text" in geom_group:
            p = p + geom_text(data=class_coord,mapping=aes(x=x_text,y=y_text,color=obj.call_.target,label=class_coord.index),**text_args_group)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add color scale
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = p + scale_color_manual(values=palette,labels=index)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set x label
    if x_label is None:
        x_label = "Can{} ({}%)".format(axis[0]+1,round(obj.eig_.iloc[axis[0],2],1))
    #set y label
    if y_label is None:
        y_label = "Can{} ({}%)".format(axis[1]+1,round(obj.eig_.iloc[axis[1],2],1))
    #set title
    if title is None:
        title = "Biplot of individuals and variables - {}".format(obj.__class__.__name__)

    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_dica(
        obj, element="biplot",**kwargs
):
    """
    Visualize Discriminant Correspondence Analysis (DiCA)

    Discriminant correspondence analysis (DiCA) is a canonical discriminant analysis on qualitative predictors. :class:`~discrimintools.plot.fviz_dica` provides plotnine based elegant visualization of DiCA outputs.
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~discrimintools.discriminant_analysis.DiCA`.

    element : str, default = 'biplot'
        The element to plot from the output, possible values: 

        - 'ind' for the individuals graphs
        - 'var' for the variables graphs
        - 'quali_var' for qualitative variables graphs
        - 'biplot' for biplot of individuals and variables
        - 'dist' for the distance graphs
    
    **kwargs : 
        further arguments passed to or from other methods

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.plot.fviz_dica_biplot`
        Visualize Discriminant Correspondence Analysis (DiCA) - Biplot of individuals and variables
    :class:`~discrimintools.plot.fviz_dica_ind`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of individuals
    :class:`~discrimintools.plot.fviz_dica_quali_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - Graph of qualitative variables
    :class:`~discrimintools.plot.fviz_dica_var`
        Visualize Discriminant Correspondence Analysis (DiCA) - graph of variables/categories
    :class:`~discrimintools.plot.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_divay
    >>> from discrimintools import DiCA, fviz_dica
    >>> D = load_divay() # load training dataset
    >>> y, X = D["Region"], D.drop(columns=["Region"]) # split into X and y
    >>> clf = DiCA()
    >>> clf.fit(X,y)
    DiCA()

    Graph of individuals...

    >>> p = fviz_dica(clf, "ind") # graph of individuals
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_ind.png
        :scale: 90%

    Graph of variables ...

    >>> p = fviz_dica(clf, "var") # graph of variables/categories
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_var.png
        :scale: 90%

    Graph of qualitative variables...

    >>> p = fviz_dica(clf, "quali_var") # graph of qualitative variables
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_quali_var.png
        :scale: 90%

    Biplot of individuals and variables...

    >>> p = fviz_dica(clf, "biplot") # biplot of individuals and variables
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_biplot.png
        :scale: 90%

    Distance between class barycenter.

    >>> p = fviz_dica(clf, "dist") # distance between barycenter
    >>> print(p)

    .. figure:: ../../../_static/fviz_dica_dist.png
        :scale: 90%

    """
    if element == "ind":
        return fviz_dica_ind(obj,**kwargs)
    elif element == "var":
        return fviz_dica_var(obj,**kwargs)
    elif element == "quali_var":
        return fviz_dica_quali_var(obj,**kwargs)
    elif element == "biplot":
        return fviz_dica_biplot(obj,**kwargs)
    elif element == "dist":
        return fviz_dist(obj,**kwargs)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'biplot', 'dist'")