
# Load dataset
from discrimintools.datasets import load_alcools, load_heart
from discrimintools import DISCRIM, STEPDISC, summarySTEPDISC, summaryDA


D = load_alcools()
y, X = D["TYPE"], D.drop(columns=["TYPE"])

clf = DISCRIM()
clf.fit(X,y)

print("\nBackward\n")
clf2 = STEPDISC(method="backward",alpha=0.01,verbose=True)
clf2.fit(clf)

print(clf2.summary_.selected)
print(clf2.disc_.coef_)

print("\nForward\n")
clf3 = STEPDISC(method="forward",alpha=0.01,verbose=True)
clf3.fit(clf)
print(clf3.summary_.selected)
print(clf3.disc_.coef_)

#training dataset
D, DTest = load_heart("train"), load_heart("test")
y, X = D["disease"], D.drop(columns=["disease"])
yTest, XTest = DTest["disease"], DTest.drop(columns=["disease"])
print(D.info())

clf = DISCRIM()
print(clf.fit(X,y))

clf2 = STEPDISC(method="backward",alpha=0.01,verbose=False)
clf2.fit(clf)
summarySTEPDISC(clf2)