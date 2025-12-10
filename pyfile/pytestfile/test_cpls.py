
from discrimintools.datasets import load_dataset, load_heart, load_alcools
from discrimintools import CPLS, summaryCPLS, fviz_plsr

D = load_dataset("breast")
y, X = D["Class"], D.drop(columns=["Class"])

clf = CPLS()
clf.fit(X,y)

summaryCPLS(clf)
summaryCPLS(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())

#training dataset
D = load_heart("train")
y, X = D["disease"], D.drop(columns=["disease"])

clf = CPLS(n_components=2)
clf.fit(X,y)

summaryCPLS(clf)
summaryCPLS(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())

clf = CPLS(n_components=2, var_select=True)
clf.fit(X,y)
summaryCPLS(clf)
summaryCPLS(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())