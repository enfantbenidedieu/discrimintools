# -*- coding: utf-8 -*-
from numpy import cos, sin, linspace, pi,sqrt,arctan2, asarray
from plotnine import aes, geom_text, theme_minimal,labs, ylim, xlim, geom_hline, geom_vline, theme, element_line, annotate

#interns functions
from discrimintools.discriminant_analysis.functions.utils import check_is_dataframe, check_is_bool
from .utils import check_is_valid_geom

#list of colors
list_colors = ["black", "red", "green", "blue", "cyan", "magenta","darkgray", "darkgoldenrod", "darkgreen", "violet",
                "turquoise", "orange", "lightpink", "lavender", "yellow","lightgreen", "lightgrey", "lightblue", "darkkhaki",
                "darkmagenta", "darkolivegreen", "lightcyan", "darkorange","darkorchid", "darkred", "darksalmon", 
                "darkseagreen","darkslateblue", "darkslategray", "darkslategrey","darkturquoise", 
                "darkviolet", "lightgray", "lightsalmon","lightyellow", "maroon"]

def add_scatter(
        p,
        data,
        axis = [0,1],
        geom = ("point","text"),
        repel = False,
        color = "steelblue",
        points_args = dict(shape = "o", size=1.5),
        text_args = dict(size=8)
):
    """
    Add elements points and/or texts to plotnine graph

    Parameters
    ----------
    p : class
        An object of class ggplot.

    data : DataFrame of shape (n_samples, n_components)
        Input data, where ``n_samples`` is the number of samples and ``n_components`` is the number of components.

    axis : list, defaul = [0,1]
        Dimensions to be plotted

    geom : str, list or tuple, default=('point','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["point","text"]. 

        - 'point' to show only points,
        - 'text' to show only labels,
        - ('point','text') to show both types.

    repel : bool, default=False 
        To avoid overplotting text labels.

    color : str, default = "steelblue"
        Color for the points and texts.

    point_args : dict, default=dict(shape = "o", size = 1.5)
        Keywords arguments for `geom_point <https://plotnine.org/reference/geom_point.html>`_.
    
    text_args : dict, default=dict(size = 8)
        KeywordS arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.

    Returns
    -------
    p : class
        A object of class ggplot.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, add_scatter
    >>> from plotnine import ggplot, theme_minimal
    >>> D = load_wine("train") # load training dataset
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> p = add_scatter(ggplot(),clf.ind_.coord,repel=True)+theme_minimal()
    >>> print(p)

    .. figure:: ../../../../_static/add_scatter.png
        
        Add individuals points - CANDISC
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if p is an instance of class ggplot
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if p.__class__.__name__ != "ggplot":
        raise TypeError("'p' must be an instance of class ggplot")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if data is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(data)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #chack if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom,("point","text"))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check is repel is a bool
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_bool(repel)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set text arguments - add overlap arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and "text" in geom:
        text_args = dict(**text_args,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))
      
    #add points
    if "point" in geom:
        p = p + annotate("point",x=asarray(data.iloc[:,axis[0]]),y=asarray(data.iloc[:,axis[1]]),color=color,**points_args)
    #add texts
    if "text" in geom:
        p = p + geom_text(data=data,mapping=aes(x = f"Can{axis[0]+1}",y=f"Can{axis[1]+1}",label=data.index),color=color,**text_args)
    return p

def set_axis(
        p,
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
    Set axis to plotnine graph

    Parameters
    ----------
    p : class
        An object of class ggplot.
    
    x_lim : None, list or tuple, default = None
        The range of the plotted ``x`` values
    
    y_lim : None, list or tuple, default = None
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

    Examples
    --------
    >>> from plotnine import ggplot
    >>> from discrimintools import set_axis
    >>> p = set_axis(ggplot())

    .. figure:: ../../../../_static/set_axis.png
        
        Set axis ggplot
    """
    #set title
    if title is None:
        title = "Map"
    p = p + labs(title=title,x=x_label,y=y_label)
    #set x limits
    if x_lim is not None:
        p = p + xlim(x_lim)
    #set y limits
    if y_lim is not None:
        p = p + ylim(y_lim)
    #add horizontal line
    if add_hline:
        p = p + geom_hline(yintercept=0,alpha=0.5,color="black",size=0.5,linetype="dashed")
    #add vertical line
    if add_vline:
        p = p+ geom_vline(xintercept=0,alpha=0.5,color="black",size=0.5,linetype="dashed")
    #add grid
    if add_grid:
        p = p + theme(panel_grid_major=element_line(alpha=None,color="black",size=0.5,linetype="dashed"))
    #set ggthme
    if ggtheme is None:
        ggtheme = theme_minimal()
    
    return p + ggtheme

def overlap_coord(
        coord,x_name,y_name,repel
):
    """
    Overlap text position for segment

    Parameters
    ----------
    coord : DataFrame of shape (n_features, n_components)
        Input data, where ``n_features`` is the number of features and ``n_components`` is the number of components.

    x_name : str
        Name of columns for `x` axis.

    y_name : str
        Name of columns for `y` axis.

    repel : bool
        To avoid overplotting text labels.

    Returns
    -------
    coord : DataFrame of shape (n_features, n_components + 2)
        Output data.

    Examples
    --------
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, overlap_coord
    >>> D = load_wine("train") # load training dataset
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(X,y)
    CANDISC()
    >>> coord = overlap_coord(clf.var_.total,"Can1","Can2",True)
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if coord is an instance of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_dataframe(coord)

    def rshift(r,theta,a=0.03,b=0.07):
        return r + a + b*abs(cos(theta))
    
    if repel:
        coord = (coord
                    .assign(
                        r = lambda x : sqrt(x[f"{x_name}"]**2+x[f"{y_name}"]**2),
                        theta = lambda x : arctan2(x[f"{y_name}"],x[f"{x_name}"]),
                        rnew = lambda x : rshift(r=x["r"],theta=x["theta"]),
                        xnew = lambda x : x["rnew"]*cos(x["theta"]),
                        ynew = lambda x : x["rnew"]*sin(x["theta"])
                        )
                    .drop(columns=["r","theta","rnew"]))
    else:
        coord = coord.assign(xnew = lambda x : x[f"{x_name}"], ynew = lambda x : x[f"{y_name}"])
    return coord 

#drwa circle to plot
def fviz_circle(
        p, r = 1.0, x0 = 0.0,  y0 = 0.0, color = "black"
):
    """
    Add a circle with plotnine

    Draw (add) a circle with plotnine based on center and radius

    Parameters
    ----------
    p : class
        An object of class ggplot.

    r : float, default = 1.0
        Radius.

    x0 : float, default = 0.0
        ``x`` center.

    y0 : float, default = 0.0
        ``y`` center.

    color : str, default = 'black'
        Color of the circle.

    Returns
    -------
    p : class
        A object of class ggplot.
    
    Examples
    --------
    >>> from plotnine import ggplot, theme_minimal
    >>> from discrimintools import fviz_circle
    >>> p = fviz_circle(ggplot()) + theme_minimal()

    .. figure:: ../../../../_static/fviz_circle.png
    
        Draw circle ggplot
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if p is an instance of class ggplot
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if p.__class__.__name__ != "ggplot":
        raise TypeError("'p' must be an instance of class ggplot")

    x = x0 + r*cos(linspace(0,pi,num=100))
    ymin, ymax = y0 + r*sin(linspace(0,-pi,num=100)), y0 + r*sin(linspace(0,pi,num=100))
    return p + annotate("ribbon", x=x, ymin=ymin, ymax=ymax,color=color,fill=None)