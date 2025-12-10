
# Load dataset
from discrimintools.datasets import load_heart
from discrimintools import MDA, summaryMDA

D = load_heart("subset")
y, X = D["disease"], D.drop(columns=["disease"])
print(D.head())

clf = MDA(n_components=5)
clf.fit(X,y)

summaryMDA(clf)
summaryMDA(clf,detailed=True)
