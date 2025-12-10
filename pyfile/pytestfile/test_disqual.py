
# Load dataset
from discrimintools.datasets import load_vote, load_canines, load_mushroom, load_oliveoil
from discrimintools import GFALDA, summaryGFALDA

D = load_vote("train")
print(D.head())
y, X = D["group"], D.drop(columns=["group"])

clf = GFALDA(n_components=5)
print(clf.fit(X,y))

summaryGFALDA(clf)
summaryGFALDA(clf,detailed=True)


D = load_oliveoil("train")
print(D.head())
D = load_canines("train")
print(D.head())
y, X = D["Fonction"], D.drop(columns=["Fonction"])

clf = GFALDA(n_components=2)
clf.fit(X,y)

summaryGFALDA(clf)
summaryGFALDA(clf,detailed=True)

D = load_mushroom("train")
print(D.head())
y, X = D["classe"], D.drop(columns=["classe"])

clf = GFALDA(n_components=5)
clf.fit(X,y)

summaryGFALDA(clf)
summaryGFALDA(clf,detailed=True)
