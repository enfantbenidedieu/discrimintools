# -*- coding: utf-8 -*-
from discrimintools.datasets import load_dataset, load_vins
from discrimintools import PLSLOGIT, fviz_plsr, summaryPLSLOGIT

D = load_dataset("breast")
y, X = D["Class"], D.drop(columns=["Class"])

clf = PLSLOGIT(n_components=2,warn_message=False)
clf.fit(X,y)
print(clf.coef_.raw.to_frame())
print(clf.decision_function(X).head())
evl = clf.eval_predict(X,y,verbose=True)
print(clf.pred_table(X,y))
print(clf.predict(X).head())
print(clf.predict_log_proba(X).head())
print(clf.predict_proba(X).head())
print(clf.score(X,y))

summaryPLSLOGIT(clf)
summaryPLSLOGIT(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())

D = load_vins("train")
y, X = D["Qualite"], D.drop(columns=["Qualite"])

clf = PLSLOGIT(classes=('Mediocre','Moyen','Bon'))
clf.fit(X,y)
print(clf.coef_.raw)
print(clf.decision_function(X).head())
evl = clf.eval_predict(X,y,verbose=True)
print(clf.pred_table(X,y))
print(clf.predict(X).head())
print(clf.predict_log_proba(X).head())
print(clf.predict_proba(X).head())
print(clf.score(X,y))

summaryPLSLOGIT(clf)
summaryPLSLOGIT(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())

clf = PLSLOGIT(multi_class="ordinal",classes=('Mediocre','Moyen','Bon'),method='bfgs')
clf.fit(X,y)
print(clf.coef_.raw)
print(clf.decision_function(X).head())
evl = clf.eval_predict(X,y,verbose=True)
print(clf.pred_table(X,y))
print(clf.predict(X).head())
print(clf.predict_log_proba(X).head())
print(clf.predict_proba(X).head())
print(clf.score(X,y))

summaryPLSLOGIT(clf)
summaryPLSLOGIT(clf,detailed=True)

p = fviz_plsr(clf,element="ind",repel=False)
print(p.show())
p = fviz_plsr(clf,element="var",repel=True)
print(p.show())
p = fviz_plsr(clf,element="dist",repel=True)
print(p.show())