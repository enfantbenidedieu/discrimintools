# Load dataset
from discrimintools.datasets import load_heart, load_alcools
from discrimintools import DISCRIM, summaryDISCRIM

D = load_heart("train")
print("\nTraining dataset (infos):")
print(D.head())
y, X = D["disease"], D.drop(columns=["disease"])

clf = DISCRIM(var_select=False)
clf.fit(X,y)
print(X.head())
print(clf.transform(X).head())

summaryDISCRIM(clf)
summaryDISCRIM(clf,detailed=True)

clf = DISCRIM(var_select=True)
clf.fit(X,y)

summaryDISCRIM(clf)
summaryDISCRIM(clf,detailed=True)

D = load_alcools("train")
print("\nTraining dataset (infos):")
print(D.head())
y, X = D["TYPE"], D.drop(columns=["TYPE"])

clf = DISCRIM()
print(clf.fit(X,y))

summaryDISCRIM(clf)
summaryDISCRIM(clf,detailed=True)


clf = DISCRIM(var_select=True)
clf.fit(X,y)

summaryDISCRIM(clf)
summaryDISCRIM(clf,detailed=True)