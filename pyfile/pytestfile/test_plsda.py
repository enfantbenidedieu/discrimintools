from discrimintools.datasets import load_dataset, load_alcools
from discrimintools import PLSDA, summaryPLSDA, fviz_plsr

#training dataset
D = load_dataset("breast")
print(D.head())
y, X = D["Class"], D.drop(columns=["Class"])

clf = PLSDA()
print(clf.fit(X,y))

summaryPLSDA(clf)
summaryPLSDA(clf,detailed=True)

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

clf = PLSDA(n_components=8)
clf.fit(X,y)

summaryPLSDA(clf)
summaryPLSDA(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())