
# Load dataset
from discrimintools.datasets import load_heart
from discrimintools import GFALDA, summaryGFALDA

D = load_heart("subset")
y, X = D["disease"], D.drop(columns=["disease"])
print(D.head())

clf = GFALDA(n_components=5)
print(clf.fit(X,y))

summaryGFALDA(clf)
summaryGFALDA(clf,detailed=True)