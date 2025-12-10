#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#title: Linear Discriminant Analysis (LDA)
#subtitle: proportional priors probabilities
#authors: Duvérier DJIFACK ZEBAZE
#email: djifacklab@gmail.com
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nLinear Discriminant Analysis with Proportional Priors Probabilities\n")

from discrimintools.datasets import load_alcools
from discrimintools import DISCRIM, summaryDISCRIM

#training dataset
D = load_alcools()
print("\nTraining data (5 first):")
print(D.head())
y, X = D["TYPE"], D.drop(columns=["TYPE"])

#instanciation
clf = DISCRIM()

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

#summary information
print("\nSummary Information:")
print(clf.summary_.infos)

#class level information
print("\nClass Level Informations:")
print(clf.classes_.infos)

#Within - class Sum of squared Cross Product (WSSCP)
print("\nWithin-Class SSCP Matrices:")
for k in clf.sscp_.within.keys():
    print("\n{} = {}".format(clf.call_.target,k))
    print(clf.sscp_.within[k].round(3))

print("\nPooled Within-Class SSCP Matrix:")
print(clf.sscp_.pooled.round(3))

print("\nBetween-Class SSCP Matrix:")
print(clf.sscp_.between.round(3))

print("\nTotal-Sample SSCP Matrix:")
print(clf.sscp_.total.round(3))

#Within-Class Covariance Matrices
print("\nWithin-Class Covariance Matrices:")
for k in clf.cov_.within.keys():
    print("\n{} = {}, DF = {}".format(clf.call_.target,k,clf.classes_.infos.loc[k,"Frequency"]-1))
    print(clf.cov_.within[k].round(3))

print("\nPooled Within-Class Covariance Matrix, DF = {}".format(clf.summary_.infos.iloc[1,3]))
print(clf.cov_.pooled.round(3))

print("\nBetween-Class Covariance Matrix, DF = {}".format(clf.summary_.infos.iloc[2,3]))
print(clf.cov_.between.round(3))

print("\nTotal-Sample Covariance Matrix, DF = {}".format(clf.summary_.infos.iloc[0,3]))
print(clf.cov_.total.round(3))

#Correlation coefficients
print("\nWithin-Class Correlation Coefficients/Pr>|r|")
for k in clf.corr_.within.keys():
    print("\n{} = {}".format(clf.call_.target,k))
    print(clf.corr_.within[k].round(3))

print("\nPooled Within-Class Correlation Coefficients/Pr>|r|")
print(clf.corr_.pooled.round(3))

print("\nBetween-Class Correlation Coefficients/Pr>|r|")
print(clf.corr_.between.round(3))

print("\nTotal-Sample Correlation Coefficients/Pr>|r|")
print(clf.corr_.total.round(3))

#summary
print("\nSimple Statistics:")
print("\nTotal-Sample")
print(clf.summary_.total.round(3))

for k in clf.summary_.within.keys():
    print("\n{} = {}".format(clf.call_.target,k))
    print(clf.summary_.within[k].round(3))

#total-sample standardized class means
print("\nTotal-Sample Standardized Class Means:")
print(clf.classes_.total.round(3))

print("\nPooled-Within Class Standardized Class Means:")
print(clf.classes_.pooled.round(3))

print("\nCovariance Matrices information:")
print(clf.cov_.infos.round(3))

print("\nSquared (Mahalanobis) Distance to {}:".format(clf.call_.target))
print(clf.classes_.mahal.round(3))

print("\nGeneralized Squared Distance to {}:".format(clf.call_.target))
print(clf.classes_.gen.round(3))

print("\nUnivariate Test Statistics:")
print(clf.statistics_.anova.round(3))

print("\nAverage R-square:")
print(clf.statistics_.average_rsq.round(3))

print("\nMultivariate Statistics and F Approximations:")
print(clf.statistics_.manova.round(3))

print("\nLinear Discriminant Function for {}:".format(clf.call_.target))
print(clf.coef_)

print("\nSummary:")
summaryDISCRIM(clf)

print("\nSummary (markdown output):")
summaryDISCRIM(clf, to_markdown=True)

print("\nSummary (all detailed):")
summaryDISCRIM(clf, detailed=True)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#evaluation of prediction on testing dataset
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nEvaluation of Prediction on Testing Dataset\n")
DTest = load_alcools("test")
print(DTest.head())
yTest, XTest = DTest["TYPE"], DTest.drop(columns=["TYPE"])

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