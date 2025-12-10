
from discrimintools.datasets import load_heart
from discrimintools import DISCRIM, eval_predict, summaryDA
D = load_heart("train")
y, X = D["disease"], D.drop(columns=["disease"])

clf = DISCRIM()
print(clf.fit(X,y))

eval_predict(clf)
summaryDA(clf)