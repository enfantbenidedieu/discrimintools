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

    Description
    -----------
    Draw the canonical discriminant analysis individuals graphs

    Usage
    -----
    ```python
    >>> fviz_candisc(self, axis = [0,1],x_label = None,y_label = None,x_lim = None,y_lim = None,title = None,geom = ["point", "text"],point_size = 1.5,text_size = 8,
                     text_type = "text", add_grid = True, add_hline = True, add_vline=True, repel = False, hline_color="black", hline_style="dashed",vline_color="black",
                     vline_style ="dashed",ha = "center",va = "center",ggtheme=pn.theme_minimal())
    ```

    Parameters
    ----------
    `self` : an object of class CANDISC

    `axis` : a numeric list/tuple of length 2 specifying the dimensions to be plotted (by default = [0,1]).

    `x_label` : a string specifying the label text of x (by default = None and a x_label is chosen).

    `y_label` : a string specifying the label text of y (by default = None and a y_label is chosen).

    `x_lim` : a numeric list of length 2 specifying the range of the plotted 'x' values (by default = None).

    `y_lim` : a numeric list of length 2 specifying the range of the plotted 'Y' values (by default = None).

    `title` : a string corresponding to the title of the graph you draw (by default = None and a title is chosen).

    `geom` : a string specifying the geometry to be used for the graph. Allowed values are the combinaison of ["point","text"]. Use "point"  (to show only points); "text" to show only labels; ["point","text"] to show both types.

    `point_size` : a numeric value specifying the marker size (by default = 1.5).
    
    `text_size` : a numeric value specifying the label size (by default = 8).

    `text_type` :  a string specifying either `geom_text` or `geom_label` (by default = "text"). Allowed values are : "text" or "label".

    `add_grid` : a boolean to either add or not a grid customization (by default = True).

    `add_hline` : a boolean to either add or not a horizontal ligne (by default = True).

    `add_vline` : a boolean to either add or not a vertical ligne (by default = True).

    `repel` : a boolean, whether to avoid overplotting text labels or not (by default == False).

    `hline_color` : a string specifying the horizontal ligne color (by default = "black").

    `hline_style` : a string specifying the horizontal ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `vline_color` : a string specifying the vertical ligne color (by default = "black").

    `vline_style` : a string specifying the vertical ligne style (by default = "dashed"). Allowed values are : "solid", "dashed", "dashdot" or "dotted"

    `ha` : horizontal alignment (by default = "center"). Allowed values are : "left", "center" or "right"

    `va` : vertical alignment (by default = "center"). Allowed values are : "top", "center", "bottom" or "baseline"

    `ggtheme`: function, plotnine theme name. Default value is theme_minimal(). Allowed values include plotnine official themes : theme_gray(), theme_bw(), theme_classic(), theme_void(),...

    Return
    ------
    a plotnine

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com

    Examples
    --------
    ```python
    >>> from seaborn import load_dataset
    >>> iris = load_dataset("iris")
    >>> from discrimintools import CANDISC, fviz_candisc
    >>> candisc = CANDISC(n_components=2,target=["species"],priors="prop",parallelize=False)
    >>> candisc.fit(iris)
    >>> p = fviz_candisc(candisc)
    >>> print(p)
    ```
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