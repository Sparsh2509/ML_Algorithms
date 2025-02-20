# seaborn providing dataset 
import seaborn as sns
# for data frames and changes 
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
## accuracy score
from sklearn.metrics import accuracy_score,classification_report

# data set iris
df=sns.load_dataset('iris')

# head shows some rows from the dataset in which features are there which used to predict the output feature here species
df.head()

# it will show the species present in the dataset which can be predict by the dataset
df['species'].unique()


df.isnull().sum()
df=df[df['species']!='setosa']
df.head()
df['species']=df['species'].map({'versicolor':0,'virginica':1})
df.head()
### Split dataset into independent and dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
# X
# y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier=LogisticRegression()

parameter={'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]}
classifier_regressor=GridSearchCV(classifier,param_grid=parameter,scoring='accuracy',cv=5)
classifier_regressor.fit(X_train,y_train)
print(classifier_regressor.best_params_)
print(classifier_regressor.best_score_)
##prediction
y_pred=classifier_regressor.predict(X_test)

score=accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))
##EDA
sns.pairplot(df,hue='species')
df.corr()