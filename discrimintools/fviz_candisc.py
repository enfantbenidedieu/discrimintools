# -*- coding: utf-8 -*-
import plotnine as pn
import pandas as pd

from .text_label import text_label

def fviz_candisc(self,
                 axis = [0,1],
                 x_label = None,
                 y_label = None,
                 x_lim = None,
                 y_lim = None,
                 title = None,
                 geom = ["point", "text"],
                 point_size = 1.5,
                 text_size = 8,
                 text_type = "text",
                 add_grid = True,
                 add_hline = True,
                 add_vline=True,
                 repel = False,
                 hline_color="black",
                 hline_style="dashed",
                 vline_color="black",
                 vline_style ="dashed",
                 ha = "center",
                 va = "center",
                 ggtheme=pn.theme_minimal()) -> pn:
    """
    Draw the Canonical Discriminant Analysis (CANDISC) individuals graphs
    ---------------------------------------------------------------------

    Author:
    ------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    if self.model_ != "candisc":
        raise ValueError("'self' must be an object of class 'CANDISC'")
    
    if ((len(axis) !=2) or 
        (axis[0] < 0) or 
        (axis[1] > self.call_["n_components"]-1)  or
        (axis[0] > axis[1])) :
        raise ValueError("You must pass a valid 'axis'.")
    
    # Initialize coordinates
    coord = self.ind_["coord"]

    # Add target variable
    coord = pd.concat((coord, self.call_["X"][self.call_["target"]]),axis=1) 

    p = pn.ggplot(data=coord,mapping=pn.aes(x = f"LD{axis[0]+1}",y=f"LD{axis[1]+1}",label=coord.index,color=self.call_["target"]))
    
    if "point" in geom:
        p = p + pn.geom_point(size=point_size)
    
    if "text" in geom:
        if repel:
            p = p + text_label(text_type,mapping=pn.aes(color=self.call_["target"]),size=text_size,va=va,ha=ha,
                               adjust_text={'arrowprops': {'arrowstyle': '-','lw':1.0}})
        else:
            p = p + text_label(text_type,mapping=pn.aes(color=self.call_["target"]),size=text_size,va=va,ha=ha)

    # Set x label
    if x_label is None:
        x_label = f"Canonical {axis[0]+1}"
    # Set y label
    if y_label is None:
        y_label = f"Canonical {axis[1]+1}"
    # Set title
    if title is None:
        title = "Canonical Discriminant Analysis"
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