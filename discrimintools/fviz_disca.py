
# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd

from .text_label import text_label

def fviz_disca_ind(self,
                    axis=[0,1],
                    x_lim = None,
                    y_lim = None,
                    x_label=None,
                    y_label=None,
                    title = None,
                    geom = ["point","text"],
                    repel=True,
                    point_size = 1.5,
                    text_size = 8,
                    text_type = "text",
                    add_grid = True,
                    add_hline = True,
                    add_vline=True,
                    ha = "center",
                    va = "center",
                    hline_color = "black",
                    hline_style = "dashed",
                    vline_color = "black",
                    vline_style = "dashed",
                    add_group = True,
                    center_marker_size=5,
                    ggtheme=pn.theme_minimal()):
    """
    Draw the Discriminant Correspondence Analysis (CANDISC) individuals graphs
    --------------------------------------------------------------------------

    Author:
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "disca":
        raise TypeError("'self' must be an object of class DISCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.factor_model_.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'")
    
    # Initialize coordinates
    coord = self.ind_["coord"]

    # Add target variable
    coord = pd.concat([coord, self.call_["X"][self.call_["target"]]],axis=1)

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index,color=self.call_["target"]))
    
    if "point" in geom:
        p = p + pn.geom_point(size=point_size)
    
    if "text" in geom:
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(color=self.call_["target"]),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=self.call_["target"]),size=text_size,va=va,ha=ha)
    
    if add_group:
        classes_coord = self.classes_["coord"]
        if "point" in geom:
            p = p + pn.geom_point(classes_coord,pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=classes_coord.index,color=classes_coord.index),size=center_marker_size)
    
   # Set x label
    if x_label is None:
        x_label = f"Canonical {axis[0]+1}"
    # Set y label
    if y_label is None:
        y_label = f"Canonical {axis[1]+1}"
    if title is None:
        title = "Individuals factor map - DISCA"
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    # Set x limits
    if x_lim is not None:
        p = p +  pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p +pn.ylim(y_lim)
    
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p

def fviz_disca_mod(self,
                 axis=[0,1],
                 x_lim= None,
                 y_lim=None,
                 x_label = None,
                 y_label = None,
                 title =None,
                 color ="black",
                 geom = ["point","text"],
                 text_type = "text",
                 marker = "o",
                 point_size = 1.5,
                 text_size = 8,
                 add_grid =True,
                 add_group=True,
                 color_sup = "blue",
                 marker_sup = "^",
                 add_hline = True,
                 add_vline = True,
                 ha="center",
                 va="center",
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 repel=False,
                 ggtheme=pn.theme_minimal()) -> pn:
    
    """
    Visualize Discriminant Correspondence Analysis - Graph of variables/categories
    ------------------------------------------------------------------------------

    Description
    -----------


    Parameters
    ----------
    self : an object of class DISCA

    axis : a numeric list or vector of length 2 specifying the dimensions to be plotted, default = [0,1]

    Return
    ------
    a plotnine graph

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    if self.model_ != "disca":
        raise ValueError("Error : 'self' must be an object of class DISCA")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.factor_model_.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")

    ###### Initialize coordinates
    coord = self.var_["coord"]

    # Initialize
    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=coord.index))

    if "point" in geom:
        p = p + pn.geom_point(pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}"),color=color,shape=marker,size=point_size)
    if "text" in geom:
        if repel:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color,"lw":1.0}})
        else:
            p = p + text_label(text_type,color=color,size=text_size,va=va,ha=ha)
    
    
    ###################### Add supplementary columns coordinates
    if add_group:
        classes_coord = self.classes_["coord"]
        if "point" in geom:
            p  = p + pn.geom_point(classes_coord,pn.aes(x=f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=classes_coord.index),
                                    color=color_sup,shape=marker_sup,size=point_size)
        if "text" in geom:
            if repel:
                p = p + text_label(text_type,data=classes_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=classes_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha,adjust_text={'arrowprops': {'arrowstyle': '-','color': color_sup,"lw":1.0}})
            else:
                p  = p + text_label(text_type,data=classes_coord,mapping=pn.aes(x = f"Dim.{axis[0]+1}",y=f"Dim.{axis[1]+1}",label=classes_coord.index),
                                    color=color_sup,size=text_size,va=va,ha=ha)
    
    # Set x label
    if x_label is None:
        x_label = f"Canonical {axis[0]+1}"
    # Set y label
    if y_label is None:
        y_label = f"Canonical {axis[1]+1}"
    if title is None:
        title = "Variables/categories factor map - DISCA"
    
    p = p + pn.labs(title = title, x = x_label, y = y_label)
    # Set x limits
    if x_lim is not None:
        p = p +  pn.xlim(x_lim)
    # Set y limits
    if y_lim is not None:
        p = p +pn.ylim(y_lim)
    
    if add_hline:
        p = p + pn.geom_hline(yintercept=0, colour=hline_color, linetype =hline_style)
    if add_vline:
        p = p+ pn.geom_vline(xintercept=0, colour=vline_color, linetype =vline_style)
    if add_grid:
        p = p + pn.theme(panel_grid_major = pn.element_line(color = "black",size = 0.5,linetype = "dashed"))

    # Add theme
    p = p + ggtheme

    return p