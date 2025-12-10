#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#title: Discriminant Correspondence Analysis (DiCA)
#subtitle: canines dataset
#authors: Duvérier DJIFACK ZEBAZE
#email: djifacklab@gmail.com
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nDiscriminant Correspondence Analysis (DiCA)\n")

from discrimintools.datasets import load_canines
from discrimintools import DiCA, summaryDiCA

#heart dataset
D = load_canines("train")
print("\nTraining data (5 first):")
print(D.head())
y, X = D["Fonction"], D.drop(columns=["Fonction"])

#instanciation
clf = DiCA()

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
summaryDiCA(clf)

print("\nSummary (markdown output):")
summaryDiCA(clf, to_markdown=True)

print("\nSummary (all detailed):")
summaryDiCA(clf, detailed=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#evaluation of prediction on testing dataset
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
DTest = load_canines("test")
print("\nTesting data (5 first):")
print(DTest.head())
yTest, XTest = DTest["Fonction"], DTest.drop(columns=["Fonction"])


#`decision_function` function
print("\n`decision_function` function:")
print(clf.decision_function(XTest).head())

print("\nClassification Summary:")
evl = clf.eval_predict(XTest,yTest,verbose=True)

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