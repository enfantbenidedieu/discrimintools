#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#title: Mixed Discriminant Analysis (MDA)
#subtitle: heart dataset
#authors: Duvérier DJIFACK ZEBAZE
#email: djifacklab@gmail.com
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nMixed Discriminant Analysis (MDA\n")

from discrimintools.datasets import load_heart
from discrimintools import MDA, summaryMDA

#heart dataset
D = load_heart("subset")
print("\nTraining data (5 first):")
print(D.head())
y, X = D["disease"], D.drop(columns=["disease"])

#instanciation
clf = MDA(n_components=5)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#fit function
clf.fit(X,y)

#`decision_function` function
print("\n`decision_function` function:")
print(clf.decision_function(X).head())

#`eval_predict` function
print("\n`eval_predict` function:")
eval_pred = clf.eval_predict(X,y,verbose=True)

#fit_transform function
print("\n`fit_transform` function:")
print(clf.fit_transform(X,y).head())

#`pred_table` function
print("\n`pred_table` function:")
print(clf.pred_table(X,y))

#`predict` function
print("\n`predict` function:")
print(clf.predict(X).head())

#`predict_log_proba` function
print("\n`predict_log_proba` function:")
print(clf.predict_log_proba(X).head())

#`predict_proba` function
print("\n`predict_proba` function:")
print(clf.predict_proba(X).head())

#`score` function
print("\n`score` function:")
print("Accurary = {}%".format(round(100*clf.score(X,y))))

#`transform` function
print("\n`transform` function:")
print(clf.transform(X).head())

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#canonical coefficients
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nCanonical coefficients:")
cancoef = clf.cancoef_
for i, k in enumerate(cancoef._fields):
    print("\n{} coefficients".format(k))
    print(cancoef[i].round(3))

print("\nSummary:")
summaryMDA(clf)

print("\nSummary (markdown output):")
summaryMDA(clf, to_markdown=True)

print("\nSummary (all detailed):")
summaryMDA(clf, detailed=True)