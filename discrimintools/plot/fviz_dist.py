# -*- coding: utf-8 -*-
from itertools import combinations
from pandas import DataFrame, concat
from plotnine import ggplot, aes, geom_point, geom_text, geom_segment, scale_color_manual, arrow

#interns functions
from .utils import check_is_valid_axis, check_is_valid_geom
from .fviz import set_axis, list_colors

def fviz_dist(
        obj,
        axis = [0,1],
        geom = ("point","text"),
        repel = False,
        point_args = dict(shape = "^", size = 3),
        text_args = dict(size = 11.5),
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
    Visualize distance between barycenter

    Parameters
    ----------
    obj : class
        An instance of class :class:`~discrimintools.discriminant_analysis.CANDISC`, :class:`~discrimintools.discriminant_analysis.DiCA`, :class:`~discrimintools.discriminant_analysis.CPLS`, :class:`~discrimintools.discriminant_analysis.PLSDA`, :class:`~discrimintools.discriminant_analysis.PLSLDA`, :class:`~discrimintools.discriminant_analysis.PLSLOGIT`.
    
    axis : list, defaul=[0,1]
        Dimensions to be plotted

    geom : str, list or tuple, default = ('point','text')
        Geometry to be used for the graph. Possible values are the combinaison of ["point","text"]. 

        - 'point' to show only points,
        - 'text' to show only labels,
        - ('point','text') to show both types.

    repel : bool, default = False 
        To avoid overplotting text labels.

    point_args : dict, default = dict(shape = "o", size = 1.5)
        Keywords arguments for `geom_point <https://plotnine.org/reference/geom_point.html>`_.
    
    text_args : dict, default = dict(size = 8)
        Keywords arguments for `geom_text <https://plotnine.org/reference/geom_text.html>`_.
        
    palette : None or list, default = None
        Color palette to be used for coloring by groups.
    
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
    >>> from discrimintools.datasets import load_wine
    >>> from discrimintools import CANDISC, fviz_dist
    >>> D = load_wine() # load training dataset
    >>> y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
    >>> clf = CANDISC()
    >>> clf.fit(XTrain,yTrain)
    CANDISC()
    >>> p = fviz_dist(clf) # graph of distance between barycenter
    >>> print(p)

    .. figure:: ../../../_static/fviz_candisc_dist.png
        :scale: 90%
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if obj is an instance of class CANDISC or DiCA
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ not in ["candisc","dica","cpls", "plsda","plslda","plslogit"]:
        raise TypeError("'obj' must be an instance of class CANDISC, DiCA, CPLS, PLSDA, PLSLDA, PLSLOGIT")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid iaxis
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_axis(obj,axis)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if valid geom
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    check_is_valid_geom(geom,('point','text'))

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set text arguments - add overlap arguments
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if repel and "text" in geom:
        text_args = dict(**text_args,adjust_text=dict(arrowprops=dict(arrowstyle='-',lw=1.0)))

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
    
    #classes coordinates
    coord = obj.classes_.coord
    coord[f"{obj.call_.target}"] = list(coord.index)
    #set x_text and y_text
    x_text, y_text = "Can"+str(axis[0]+1), "Can"+str(axis[1]+1)

    if obj.model_ == "candisc":
        #square mahalanobis distance
        dist2 = obj.classes_.mahal
    else:
        #squared euclidean distance
        dist2 = obj.classes_.eucl

    #all combinations of two classes
    all_comb = combinations(coord.index,r=2)
    data = DataFrame()
    i = 0
    for comb in all_comb:
        From, To = comb[0], comb[1]
        dist = round(dist2.loc[From,To],2)
        x, y = coord.loc[From,x_text], coord.loc[From,y_text]
        xend, yend = coord.loc[To,x_text], coord.loc[To,y_text]
        xmid, ymid = 0.5*(x + xend), 0.5*(y + yend)
        row = DataFrame(dict(x=x,y=y,xend=xend,yend=yend,xmid=xmid,ymid=ymid,dist=dist),index=[i])
        data = concat((data,row),axis=0)
        i +=1
    
    #initialize
    p = ggplot()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add classes coordinates
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "point" in geom:
        p = p + geom_point(data=coord,mapping=aes(x=x_text,y=y_text,color=obj.call_.target,label=coord.index),**point_args)
    if "text" in geom:
        p = p + geom_text(data=coord,mapping=aes(x=x_text,y=y_text,color=obj.call_.target,label=coord.index),**text_args)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add color scale
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = p + scale_color_manual(values=palette,labels=index)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add distance
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = p + geom_segment(data=data,mapping=aes(x="x",y="y",xend="xend",yend="yend"),color="black",linetype="dashed",size=0.5,
                         arrow = arrow(angle=30,length=0.2/2.54,ends="both",type="open"))
    #add labels
    p = p + geom_text(data=data,mapping=aes(x="xmid",y="ymid",label="dist"),color="blue",size=10)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #add others elements
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if obj.model_ in ["candisc","dica"]:
        prop = obj.eig_.iloc[:,2]
    else:
        prop = obj.explained_variance_.iloc[:,0]
    
    #set x label
    if x_label is None:
        x_label = "Can{} ({}%)".format(axis[0]+1,round(prop[axis[0]],1))
    #set y label
    if y_label is None:
        y_label = "Can{} ({}%)".format(axis[1]+1,round(prop[axis[1]],1))
    #set title
    if title is None:
        title = "Distance between barycenter - {}".format(obj.__class__.__name__)
    p = set_axis(p=p,x_lim=x_lim,y_lim=y_lim,x_label=x_label,y_label=y_label,title=title,add_hline=add_hline,add_vline=add_vline,add_grid=add_grid,ggtheme=ggtheme)

    return p