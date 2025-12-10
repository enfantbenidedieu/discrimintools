#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#title: Stepwise Discriminant Analysis (STEPWISE)
#subtitle: forward selection
#authors: Duvérier DJIFACK ZEBAZE
#email: djifacklab@gmail.com
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nStepwise Discriminant Analysis with forward selection\n")

from discrimintools.datasets import load_dataset
from discrimintools import DISCRIM, STEPDISC, summarySTEPDISC

#training data
D = load_dataset("wavenoise")
print("\nTraining data (5 first):")
print(D.head())
y, X = D["classe"], D.drop(columns=["classe"])

#linear discriminant analysis
clf = DISCRIM().fit(X,y)

#stepwise discriminant analysis
clf2 = STEPDISC(method="forward")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#fit function
clf2.fit(clf)

#`decision_function` function
print("\n`decision_function` function:")
print(clf2.decision_function(X).head())

#`eval_predict` function
print("\n`eval_predict` function:")
eval_pred = clf2.eval_predict(X,y,verbose=True)

#fit_transform function
print("\n`fit_transform` function:")
print(clf2.fit_transform(clf).head())

#`pred_table` function
print("\n`pred_table` function:")
print(clf2.pred_table(X,y))

#`predict` function
print("\n`predict` function:")
print(clf2.predict(X).head())

#`predict_log_proba` function
print("\n`predict_log_proba` function:")
print(clf2.predict_log_proba(X).head())

#`predict_proba` function
print("\n`predict_proba` function:")
print(clf2.predict_proba(X).head())

#`score` function
print("\n`score` function:")
print("Accurary = {}%".format(round(100*clf2.score(X,y))))

#`transform` function
print("\n`transform` function:")
print(clf2.transform(X).head())

print("\nSummary:")
summarySTEPDISC(clf2)

print("\nSummary (markdown output):")
summarySTEPDISC(clf2, to_markdown=True)

print("\nSummary (all detailed):")
summarySTEPDISC(clf2, detailed=True)