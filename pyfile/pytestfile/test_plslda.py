# -*- coding: utf-8 -*-
from discrimintools.datasets import load_dataset, load_alcools
from discrimintools import PLSLDA, summaryPLSLDA, fviz_plsr

D = load_dataset("breast")
y, X = D["Class"], D.drop(columns=["Class"])

clf = PLSLDA()
print(clf.fit(X,y))

summaryPLSLDA(clf)
summaryPLSLDA(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())

#training dataset
D = load_alcools()
print(D.head())
y, X = D["TYPE"], D.drop(columns=["TYPE"])

clf = PLSLDA(n_components=8)
clf.fit(X,y)

summaryPLSLDA(clf)
summaryPLSLDA(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())

clf = PLSLDA(n_components=4)
clf.fit(X,y)

summaryPLSLDA(clf)
summaryPLSLDA(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())