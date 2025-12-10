
from discrimintools.datasets import load_wine, load_heart, load_dataset
from discrimintools import CANDISC, fviz_candisc, summaryCANDISC

#training dataset
D, XTest = load_wine(), load_wine("test")
print(D.head())
y, X = D["Quality"], D.drop(columns=["Quality"])

clf = CANDISC()
clf.fit(X,y)

summaryCANDISC(clf)
summaryCANDISC(clf,detailed=True)

p = fviz_candisc(clf,element="ind",repel=True)
p.show()
p = fviz_candisc(clf,element="var",repel=True)
print(p.show())
p = fviz_candisc(clf,element="biplot",repel=True)
print(p.show())
p = fviz_candisc(clf,element="dist",repel=True)
print(p.show())

D = load_dataset("iris")
print(D.head())
y, X = D["species"], D.drop(columns=["species"])

clf = CANDISC()
clf.fit(X,y)
summaryCANDISC(clf)
summaryCANDISC(clf,detailed=True)

p = fviz_candisc(clf,element="ind",repel=True)
print(p.show())
p = fviz_candisc(clf,element="var",repel=True)
print(p.show())
p = fviz_candisc(clf,element="biplot",repel=True)
print(p.show())
p = fviz_candisc(clf,element="dist",repel=True)
print(p.show())

#heart dataset
D = load_heart("train")
y, X = D["disease"], D.drop(columns=["disease"])
print(D.info())

clf = CANDISC(n_components=5)
clf.fit(X,y)

summaryCANDISC(clf)
summaryCANDISC(clf,detailed=True)