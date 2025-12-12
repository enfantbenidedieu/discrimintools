# -*- coding: utf-8 -*-
from numpy import asarray
from pandas import concat
from plotnine import ggplot, aes, geom_point, geom_text, geom_segment, scale_color_manual, arrow, annotate

#interns functions
from .utils import check_is_valid_axis, check_is_valid_geom
from .fviz import overlap_coord, fviz_circle, set_axis, list_colors
from .fviz_dist import fviz_dist

def fviz_candisc_ind(
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
    Visualize Canonical Discriminant Analysis (CANDISC) - Graph of individuals

    Canonical discriminant analysis is a dimension-reduction technique related to principal component analysis and canonical correlation. :class:`~discrimintools.fviz_candisc_ind` provides plotnine based elegant visualization of CANDISC outputs for individuals.

    Parameters
    ----------
    obj : class
        An instance of class :class:`~discrimintools.CANDISC`.
    
    axis : list, defaul=[0,1]
        Dimensions to be plotted.

    geom_ind : str, list or tuple, default=('point','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["point","text"]. 

        - 'point' to show only points,
        - 'text' to show only labels,
        - ('point','text') to show both types.

    repel : bool, default=False 
        To avoid overplotting text labels.

    point_args_ind : dict, default=dict(shape = "o", size = 1.5)
        Keywords arguments for `geom_point <https://plotnine.org/reference/geom_point.html>`_.
    
    text_args_ind : dict, default=dict(size = 8)
        Keywords arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.

    add_group : bool, default = True
        To show group coordinates.

    geom_group : str, list or tuple, default = ('point','text')
        See ``geom_ind``.

    point_args_group : dict, default = dict(shape = "^", size = 3)
        See ``point_args_ind``.
        
    text_args_group : dict, default=dict(size = 11.5)
        See ``text_args_ind``.
    
    palette : None or list, default=None
        Color palette to be used for coloring by groups.
    
    x_lim : None, list or tuple, default=None
        The range of the plotted ``x`` values.
    
    y_lim : None, list or tuple, default=None
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
    
    add_grid : bool, default=True
        To add grid customization.
        
    ggtheme : function, default = None
        Plotnine `theme <https://plotnine.org/guide/themes-premade.html>`_ name.

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.fviz_candisc`
        Visualize Canonical Discriminant Analysis (CANDISC).
    :class:`~discrimintools.fviz_candisc_biplot`
        Visualize Canonical Discriminant Analysis (CANDISC) - Biplot of individuals and variables.
    :class:`~discrimintools.fviz_candisc_var`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of variables.
    :class:`~discrimintools.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, fviz_candisc_ind
    >>> D = load_wine("train") # load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> p = fviz_candisc_ind(clf) # graph of individuals
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_ind.png
    
        Graph of individuals - CANDISC
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class CANDISC
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "candisc":
        raise TypeError("'obj' must be an instance of class CANDISC")
    
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

    #concatenate individuals coordinates with target variable
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
            p = p + geom_point(data=class_coord,mapping=aes(x=x_text,y=y_text,color=obj.call_.target,label=class_coord.index),**point_args_group)
        #add texts
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
        title = "Graph of individuals - {}".format(obj.__class__.__name__)
    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_candisc_var(
        obj,
        axis = [0,1],
        geom = ("arrow","text"),
        repel = False,
        segment_args = dict(linetype="solid",size=0.5,alpha=1),
        text_args = dict(size=8),
        palette = None,
        add_circle = True,
        col_circle = "gray",
        x_lim = (-1.1,1.1),
        y_lim = (-1.1,1.1),
        x_label = None,
        y_label = None,
        title = None,
        add_hline = True,
        add_vline = True,
        add_grid = True,
        ggtheme = None
):  
    """
    Visualize Canonical Discriminant Analysis (CANDISC) - Graph of variables

    Canonical discriminant analysis is a dimension-reduction technique related to principal component analysis and canonical correlation. :class:`~discrimintools.fviz_candisc_var` provides plotnine based elegant visualization of CANDISC outputs for variables.
    
    Parameters
    ----------
    obj : class
        An instance of class :class:`~discrimintools.CANDISC`.
    
    axis : list, defaul = [0,1]
        Dimensions to be plotted

    geom : str, list or tuple, default = ('arrow','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["arrow","text"].

        - 'arrow' to show only arrows,
        - 'text' to show only labels,
        - ('arrow','text') to show both types.

    repel : bool, default = False 
        To avoid overplotting text labels.

    segments_args : dict, default = dict(linetype="solid",size=0.5,alpha=1)
        Keywords arguments for `geom_segment <https://plotnine.org/reference/geom_segment.html>`_.
    
    text_args : dict, default = dict(size = 8)
        Keywords arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.
    
    palette : None or list, default = None
        Color palette to be used for coloring by groups.

    add_circle : bool, default = True
        To draw circle.
    
    col_circle : str, default = "gray"
        Color for the circle

    x_lim : None, list or tuple, default = (-1.1,1.1)
        The range of the plotted ``x`` values
    
    y_lim : None, list or tuple, default = (-1.1,1.1)
        The range of the plotted ``y`` values
    
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
    :class:`~discrimintools.fviz_candisc`
        Visualize Canonical Discriminant Analysis (CANDISC).
    :class:`~discrimintools.fviz_candisc_biplot`
        Visualize Canonical Discriminant Analysis (CANDISC) - Biplot of individuals and variables.
    :class:`~discrimintools.fviz_candisc_ind`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of individuals.
    :class:`~discrimintools.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, fviz_candisc_var
    >>> D = load_wine("train") #load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> p = fviz_candisc_var(clf) # graph of variables
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_var.png

        Graph of variables - CANDISC
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class CANDISC
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "candisc":
        raise TypeError("'obj' must be an instance of class CANDISC")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid axis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom,choice=("arrow","text"))
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #concatenate all variables correlations : total, pooled and between
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    tcorr, pcorr, bcorr = obj.var_.total.assign(group="Total"),obj.var_.pooled.assign(group="Pooled"), obj.var_.between.assign(group="Between")
    #concatenate
    coord = concat((tcorr,pcorr,bcorr),axis=0)

    #define text coordinates - overlap texts
    coord = overlap_coord(coord=coord,x_name="Can"+str(axis[0]+1),y_name="Can"+str(axis[1]+1),repel=repel)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index and palette
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set index
    index = ["Between","Pooled","Total"]
    #set palette
    if palette is None:
        palette = list_colors[:len(index)]
    elif not isinstance(palette,(list,tuple)):
        raise TypeError("'palette' must be a list or a tuple of colors")
    elif len(palette) != len(index):
        raise TypeError(f"'palette' must be a list or tuple with length {len(index)}.")

    #initialize
    p = ggplot(data=coord,mapping=aes(x=f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",label=coord.index))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add arrows
    if "arrow" in geom:
        p = p + geom_segment(aes(x=0,y=0,xend=f"Can{axis[0]+1}",yend=f"Can{axis[1]+1}",color="group"),**segment_args, arrow = arrow(angle=30,length=0.2/2.54,type="open"))
    #dd texts
    if "text" in geom:
        p = p + geom_text(aes(x="xnew",y="ynew",color="group"),**text_args)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #scale color manual
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = p + scale_color_manual(values=palette,labels=index)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add correlation circle
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if add_circle:
        p = fviz_circle(p=p,color=col_circle)

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
        title = "Graph of variables - {}".format(obj.__class__.__name__)
    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p

def fviz_candisc_biplot(
        obj,
        axis = [0,1],
        geom_ind = ("point","text"),
        repel = False,
        point_args_ind = dict(shape= "o", size = 1.5),
        text_args_ind = dict(size=8),
        geom_var = ("arrow","text"),
        col_var = "steelblue",
        segment_args = dict(linetype="solid", size = 0.5, alpha=1),
        text_args_var = dict(size=8),
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
    Visualize Canonical Discriminant Analysis (CANDISC) - Biplot of individuals and variables

    Canonical discriminant analysis is a dimension-reduction technique related to principal component analysis and canonical correlation. :class:`~discrimintools.fviz_candisc_biplot` provides plotnine based elegant visualization of CANDISC outputs for individuals and variables.

    Parameters
    ----------
    obj : class
        An object of class :class:`~discrimintools.CANDISC`.

    **kwargs : 
        further arguments passed to or from others functions. See :class:`~discrimintools.fviz_candisc_ind`, :class:`~discrimintools.fviz_candisc_var`.

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.fviz_candisc`
        Visualize Canonical Discriminant Analysis (CANDISC).
    :class:`~discrimintools.fviz_candisc_ind`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of individuals.
    :class:`~discrimintools.fviz_candisc_var`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of variables.
    :class:`~discrimintools.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, fviz_candisc_biplot
    >>> D = load_wine("train") # load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> p = fviz_candisc_biplot(clf) # biplot of individuals and variables
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_biplot.png

        Biplot of individuals and variables - CANDISC
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class CANDISC
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ != "candisc":
        raise TypeError("'obj' must be an instance of class CANDISC")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid iaxis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom_ind,('point','text'))
    check_is_valid_geom(geom_var,('arrow','text'))

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
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #rescale variables coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    xscale = (max(obj.ind_.coord.iloc[:,axis[0]])-min(obj.ind_.coord.iloc[:,axis[0]]))/(max(obj.var_.total.iloc[:,axis[0]])-min(obj.var_.total.iloc[:,axis[0]]))
    yscale = (max(obj.ind_.coord.iloc[:,axis[1]])-min(obj.ind_.coord.iloc[:,axis[1]]))/(max(obj.var_.total.iloc[:,axis[1]])-min(obj.var_.total.iloc[:,axis[1]]))
    scale = min(xscale, yscale)

    #create coordinate
    coord = concat((obj.ind_.coord,obj.call_.y),axis=1)
    #initialize
    p = ggplot(data=coord,mapping=aes(x="Can"+str(axis[0]+1), y="Can"+str(axis[1]+1),label=coord.index))

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
    #define text coordinates - overlap texts
    var_coord = overlap_coord(coord=obj.var_.total.mul(scale),x_name="Can"+str(axis[0]+1),y_name="Can"+str(axis[1]+1),repel=repel)

    #add arrows
    if "arrow" in geom_var:
        p = p + annotate("segment",x=0,y=0,xend=asarray(var_coord.iloc[:,axis[0]]),yend=asarray(var_coord.iloc[:,axis[1]]),color=col_var,arrow = arrow(angle=30,length=0.2/2.54,type="open"),**segment_args)
    #dd texts
    if "text" in geom_var:
        p = p + geom_text(data=var_coord,mapping=aes(x="xnew",y="ynew",label=var_coord.index),color=col_var,**text_args_var)
    
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
            p = p + geom_point(data=class_coord,mapping=aes(x="Can"+str(axis[0]+1),y="Can"+str(axis[1]+1),color=obj.call_.target,label=class_coord.index),**point_args_group)
        #add texts
        if "text" in geom_group:
            p = p + geom_text(data=class_coord,mapping=aes(x="Can"+str(axis[0]+1),y="Can"+str(axis[1]+1),color=obj.call_.target,label=class_coord.index),**text_args_group)

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

def fviz_candisc(
        obj, element="biplot",**kwargs
):
    """
    Visualize Canonical Discriminant Analysis (CANDISC)

    Canonical discriminant analysis is a dimension-reduction technique related to principal component analysis and canonical correlation. :class:`~discrimintools.fviz_candisc` provides plotnine based elegant visualization of CANDISC outputs.
    
    Parameters
    ----------
    obj : class
        An object of class :class:`~discrimintools.CANDISC`.

    element : str, default = 'biplot'
        The element to plot from the output, possible values:

        - 'ind' for the individuals graphs
        - 'var' for the variables graphs (= Correlation circle)
        - 'biplot' for biplot of individuals and variables
        - 'dist' for the distance graphs
    
    **kwargs : 
        further arguments passed to or from other functions.

    Returns
    -------
    p : class
        A object of class ggplot.

    See also
    --------
    :class:`~discrimintools.fviz_candisc_biplot`
        Visualize Canonical Discriminant Analysis (CANDISC) - Biplot of individuals and variables.
    :class:`~discrimintools.fviz_candisc_ind`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of individuals.
    :class:`~discrimintools.fviz_candisc_var`
        Visualize Canonical Discriminant Analysis (CANDISC) - Graph of variables.
    :class:`~discrimintools.fviz_dist`
        Visualize distance between barycenter.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, fviz_candisc
    >>> D = load_wine("train") # load training data
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()

    Graph of individuals...

    >>> p = fviz_candisc(clf, "ind") # graph of individuals
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_ind.png

        Graph of individuals - CANDISC

    Graph of variables...
        
    >>> p = fviz_candisc(clf, "var") # graph of variables
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_var.png

        Graph of variables - CANDISC

    Biplot of individuals and variables...

    >>> p = fviz_candisc(clf, "biplot") # biplot of individuals and variables
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_biplot.png
        Biplot of individuals and variables - CANDISC

    Distance between class barycenter.

    >>> p = fviz_candisc(clf, "dist") # graph of distance
    >>> print(p)

    .. figure:: ../../../../_static/fviz_candisc_dist.png

        Distance between class barycenter - CANDISC

    """
    if element == "ind":
        return fviz_candisc_ind(obj,**kwargs)
    elif element == "var":
        return fviz_candisc_var(obj,**kwargs)
    elif element == "biplot":
        return fviz_candisc_biplot(obj,**kwargs)
    elif element == "dist":
        return fviz_dist(obj,**kwargs)
    else:
        raise ValueError("'element' should be one of 'ind', 'var', 'biplot', 'dist'")