
from discrimintools.datasets import load_divay, load_canines
from discrimintools import DiCA, fviz_dica, summaryDiCA

#divay
D = load_divay()
print(D.head())
y, X = D["Region"],D.drop(columns=["Region"])

clf = DiCA()
clf.fit(X,y)

summaryDiCA(clf)
summaryDiCA(clf,detailed=True)

p = fviz_dica(clf,element="ind",repel=True,x_lim=(-1.5,1.5),y_lim=(-1.5,1.5))
print(p.show())
p = fviz_dica(clf,element="var",repel=True,x_lim=(-1.5,1.5),y_lim=(-1.5,1.5))
print(p.show())
p = fviz_dica(clf,element="quali_var",repel=True)
print(p.show())
p = fviz_dica(clf,element="biplot",repel=True,x_lim=(-1.5,1.5),y_lim=(-1.5,1.5))
print(p.show())
p = fviz_dica(clf,element="dist",repel=True,x_lim=(-1,1),y_lim=(-1,1))
print(p.show())


#canines
D = load_canines()
print(D.head())
y,X =D["Fonction"],D.drop(columns=["Fonction"])

clf = DiCA()
print(clf.fit(X,y))

summaryDiCA(clf)
summaryDiCA(clf,detailed=True)

p = fviz_dica(clf,element="ind",repel=True,x_lim=(-1.5,1.5),y_lim=(-1.5,1.5))
print(p.show())
p = fviz_dica(clf,element="var",repel=True,x_lim=(-1.5,1.5),y_lim=(-1.5,1.5))
print(p.show())
p = fviz_dica(clf,element="biplot",repel=True,x_lim=(-1.5,1.5),y_lim=(-1.5,1.5))
print(p.show())
p = fviz_dica(clf,element="dist",repel=True,x_lim=(-1,1),y_lim=(-1,1))
print(p.show())