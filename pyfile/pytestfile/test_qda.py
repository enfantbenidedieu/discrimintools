

from discrimintools.datasets import load_alcools
from discrimintools import DISCRIM, summaryDISCRIM

#training dataset
D = load_alcools()
print(D.head())
y, X = D["TYPE"], D.drop(columns=["TYPE"])

#instanciation
clf = DISCRIM(method="quad")
clf.fit(X,y)

print("\nSummary without all informations:")
summaryDISCRIM(clf)
summaryDISCRIM(clf,to_markdown=True)

summaryDISCRIM(clf,detailed=True)
summaryDISCRIM(clf,detailed=True,to_markdown=True)