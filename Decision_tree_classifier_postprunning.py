import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris=load_iris()

# Prints whole data
print(iris.data)

# Prints target array
print(iris.target)

# Show whole data in tabluar form
df=sns.load_dataset('iris')
# Only prints input features
data=df.head()
print(data)

# X have all data input features
X=df.iloc[:,:-1]
# y has only output feature
y=iris.target

print(X)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train)


## Postpruning 
# Post prunning me yehe dikat aati hai ke vo split karta rahta hai jab tak leaf node na mil jaye par hame to next split me dikh raha hai 
# ke vo split hone ke baad majority ek output feature ko he de raha hai to obvious hai ke usse he aage split hoga to hame aage split nahi karna
# vohi tak ruk jana hai isliye ham max_depth ka use karenge 


# post prunning without max_depth
# score = 0.96
######treemodel=DecisionTreeClassifier() 


#isme 2nd level par he model ruk jayega  
# score 0.98
treemodel=DecisionTreeClassifier(max_depth= 2) 
treemodel.fit(X_train,y_train)
 
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)
plt.show()

y_pred= treemodel.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(y_pred,y_test)
print("Accuracy : ",score)

print(classification_report(y_pred,y_test))