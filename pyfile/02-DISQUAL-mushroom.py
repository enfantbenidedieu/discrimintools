#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#title: Discriminant Analysis on Qualitative Preditors (DISQUAL)
#subtitle: proportional priors probabilities
#authors: Duvérier DJIFACK ZEBAZE
#email: djifacklab@gmail.com
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nDiscriminant Analysis of Qualitative Predictors with Proportional Priors Probabilities\n")

from discrimintools.datasets import load_mushroom
from discrimintools import GFALDA, summaryGFALDA

#heart data
D = load_mushroom("train")
print("\nTraining data (5 first):")
print(D.head())
y, X = D["classe"], D.drop(columns=["classe"])

#instanciation
clf = GFALDA(n_components=2)

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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#discriminant analysis on qualitative predictors coefficients
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nDISQUAL coefficients:")
coef = clf.coef_
for i, k in enumerate(coef._fields):
    print("\n{} coefficients".format(k))
    print(coef[i].round(3))

print("\nSummary:")
summaryGFALDA(clf)

print("\nSummary (markdown output):")
summaryGFALDA(clf, to_markdown=True)

print("\nSummary (all detailed):")
summaryGFALDA(clf, detailed=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#evaluation of prediction on testing dataset
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nEvaluation of Prediction on Testing Dataset\n")
DTest = load_mushroom("test")
print(DTest.head())
yTest, XTest = DTest["classe"], DTest.drop(columns=["classe"])

#`decision_function` function
print("\n`decision_function` function:")
print(clf.decision_function(XTest).head())

#`predict` function
print("\n`predict` function:")
print(clf.predict(XTest).head())

#`predict_log_proba` function
print("\n`predict_log_proba` function:")
print(clf.predict_log_proba(XTest).head())

#`predict_proba` function
print("\n`predict_proba` function:")
print(clf.predict_proba(XTest).head())

#`transform` function
print("\n`transform` function:")
print(clf.transform(XTest).head())

#`eval_predict` function
print("\n`eval_predict` function:")
eval_test = clf.eval_predict(XTest,yTest,verbose=True)