# seaborn providing dataset 
import seaborn as sns
# for data frames and changes 
import pandas as pd
import matplotlib.pyplot as plt

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

# for checking null values in the dataset
df.isnull().sum()

# setosa removed from dataset for binary classification provide false and true value...false in place of setosa and 
# true in place of other species
# inner df is for removing setosa and outer df for showing other species
df=df[df['species']!='setosa']

# shows rows with output species
df.head()

# assigning maths number 0 and 1 to species
df['species']=df['species'].map({'versicolor':0,'virginica':1})

# now output change to 0 and 1
df.head()

### Split dataset into independent and dependent features
# In X we take all column except last one (species)
# In y we take species
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
# X
# y

# Training and Testing model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initializing logistic regression model
classifier=LogisticRegression()

# Changing in the parameter of the logistic regression  after importing gridsearchcv for tuning
# with model to show the accurate data
parameter={'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]}

# For hyper parameter tuning now use gridsearchcv
# cv = cross validation (internal divide karega dataset to check ke sabse best parameter konsa hai)
classifier_regressor=GridSearchCV(classifier,param_grid=parameter,scoring='accuracy',cv=5)

classifier_regressor.fit(X_train,y_train)

# To print the best parameter after training the model
print(classifier_regressor.best_params_)

# To print the best score(accuracy)
print(classifier_regressor.best_score_)

##prediction
y_pred=classifier_regressor.predict(X_test)

# In score we store the test accuracy
score=accuracy_score(y_pred,y_test)
print(score)

print(classification_report(y_pred,y_test))
##EDA
# capture from df and hue contain category 
a=sns.pairplot(df,hue='species')
# checking correlation if correction is postive that means accuracy is high
b=df.corr()
print(a)
print(b)
plt.show()