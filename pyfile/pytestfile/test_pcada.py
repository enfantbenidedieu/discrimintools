

# Load dataset
from discrimintools.datasets import load_alcools
from discrimintools import GFALDA, summaryGFALDA

D = load_alcools()
print(D.head())
y, X = D["TYPE"], D.drop(columns=["TYPE"])

clf = GFALDA()
print(clf.fit(X,y))

summaryGFALDA(clf)

summaryGFALDA(clf,detailed=True)