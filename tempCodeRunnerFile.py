y_pred=treemodel.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_pred,y_test)
print(score)

print(classification_report(y_pred,y_test))