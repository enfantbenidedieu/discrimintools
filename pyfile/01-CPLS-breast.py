#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#title: Partial Least Squares for Classification (CPLS)
#subtitle: breast dataset 
#authors: Duvérier DJIFACK ZEBAZE
#email: djifacklab@gmail.com
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nPartial Least Squares for Classification\n")

from discrimintools.datasets import load_dataset
from discrimintools import CPLS, summaryCPLS

#training dataset
D = load_dataset("breast")
print("\nTraining data (5 first):")
print(D.head())
y, X = D["Class"], D.drop(columns=["Class"])

#instanciation
clf = CPLS()

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

#`fit_transform` function
print("\n`fit_transform` function:")
print(clf.fit_transform(X,y).head())

#`pred_table` function
print("\n`pred_table` function:")
print(clf.pred_table(X,y))

#`predict` function
print("\n`predict` function:")
print(clf.predict(X).head())

#`score` function
print("\n`score` function:")
print("Accurary = {}%".format(round(100*clf.score(X,y))))

#`transform` function
print("\n`transform` function:")
print(clf.transform(X).head())

#classes coordinates
print("\nClass coordinates:")
print(clf.classes_.coord)

#class squares euclidean distance
print("\nClass square euclidean distance to origin:")
print(clf.classes_.eucl)

#class generalized squares distance 
print("\nClass generalized squares distance to origin:")
print(clf.classes_.gen)

#coefficients of linear model
print("\nCoefficients of linear model:")
print(clf.coef_.to_frame())

#explaiend variance
print("\nExplained variance:")
print(clf.explained_variance_)

#variable loadings
print("\nVariable loadings")
print(clf.var_.loadings)

#variable rotations
print("\nVariable rotations")
print(clf.var_.rotations)

#variable left singular matrix
print("\nVariable left singular matrix")
print(clf.var_.weights)

#variables importance in projection
print("\nVariables importance in projection:")
print(clf.vip_.vip.to_frame())

print("\nSummary:")
summaryCPLS(clf)

print("\nSummary (markdown output):")
summaryCPLS(clf, to_markdown=True)

print("\nSummary (all detailed):")
summaryCPLS(clf, detailed=True)