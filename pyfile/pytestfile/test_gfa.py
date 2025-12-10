
from discrimintools.datasets import load_alcools, load_canines, load_heart
from discrimintools import GFA, MPCA, DISCRIM, summaryGFA, summaryMPCA

print("\nPCA\n") 
D = load_alcools("train")
X = D.drop(columns=["TYPE"])
clf = GFA()
clf.fit(X)
summaryGFA(clf)

print(clf.eig_)
print("\nFit transform:")
print(clf.fit_transform(X).head())
print("\nTransform:")
print(clf.transform(X).head())

print("\nMCA\n")
D = load_canines("train")
X = D.drop(columns=["Fonction"])
clf = GFA()
clf.fit(X)
summaryGFA(clf)

print(clf.eig_)
print("\nFit transform:")
print(clf.fit_transform(X).head())
print("\nTransform:")
print(clf.transform(X).head())

print("\nFAMD\n")
D = load_heart("subset")
X = D.drop(columns=["disease"])
clf = GFA()
clf.fit(X)
summaryGFA(clf)

print(clf.eig_)
print("\nFit transform:")
print(clf.fit_transform(X).head())
print("\nTransform:")
print(clf.transform(X).head())

print("\nMPCA\n")
clf = MPCA()
clf.fit(X)
summaryMPCA(clf)

print(clf.eig_)
print("\nFit transform:")
print(clf.fit_transform(X).head())
print("\nTransform:")
print(clf.transform(X).head())