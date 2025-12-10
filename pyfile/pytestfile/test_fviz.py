from plotnine import ggplot, theme_minimal
from discrimintools import fviz_circle
p = fviz_circle(p=ggplot()) + theme_minimal()
print(p.show())

from plotnine import ggplot
from discrimintools import set_axis
p = set_axis(p=ggplot())
print(p.show())

from discrimintools.datasets import load_wine
from discrimintools import CANDISC, add_scatter
from plotnine import ggplot, theme_minimal

D = load_wine("train") # load training dataset
y, X = D["Quality"], D.drop(columns=["Quality"]) # split into X and y
clf = CANDISC()
clf.fit(X,y)
p = add_scatter(ggplot(),clf.ind_.coord,repel=True)+theme_minimal()
print(p.show())