
import altair as alt
from discrimintools.datasets import load_wine, load_heart, load_dataset
from discrimintools import CANDISC, fviz_candisc, summaryCANDISC

#training dataset
D, XTest = load_wine(), load_wine("test")
print(D.head())
y, X = D["Quality"], D.drop(columns=["Quality"])

clf = CANDISC()
clf.fit(X,y)


from pandas import concat
coord = concat((clf.ind_.coord,clf.call_.y),axis=1)
print(coord.head())

points = alt.Chart(coord).mark_point().encode(
    x='Can1',
    y='Can2',
    color="Quality"
)
points.show()
