from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Storing dataset into df
df=fetch_california_housing()
print(df)

# dataset gives row and columns of data
dataset=pd.DataFrame(df.data)
print(dataset)

# only print columns of data set or feature names
dataset.columns=df.feature_names
print(dataset.columns)

# rows and columns of dataset with features as columns name
set=dataset.head()
print(set)

## Independent features and dependent features
# X contains whole dataset and y contains only target feature(price of house) from the dataset
X=dataset
y=df.target
print(X)
print(y)

#train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(X_train)
print(y_train)

## standardizing the dataset which use to normalize the linear regression and it should be done before fit
scaler = StandardScaler()

# X_train is transforming using fit_transform and X_test is transforming using transform
# we standardize to make better output model
# fit_transform is used in train data to cal mean and standard deviation(necessary statistics) and
# Used on the training data to learn and apply the transformation in parameters of scaler

# transform is only applies the transformation using the previously learned statistics or
# we can say used on new/unseen data (e.g., test data) after the transformation has been learned from the training data.
X_train=scaler.fit_transform(X_train)
print(X_train)
X_test=scaler.transform(X_test)
print(X_test)

# To inverse the X_train we can use inverse transform
# X_train=scaler.inverse_transform(X_train)
# print(X_train)


 
from sklearn.linear_model import LinearRegression
##cross validation
from sklearn.model_selection import cross_val_score
regression=LinearRegression()
regression.fit(X_train,y_train)




#cv means cross validation which internally train and divide data and provide different accuracy and then mean accuracy
# mse provides means square error values
mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
print(mse)
# ṃean of mse
mean=-(np.mean(mse))
print(mean)

##prediction 
reg_pred=regression.predict(X_test)
print(reg_pred)


score=r2_score(reg_pred,y_test)
print(score)

# To improve r2 i used pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('model', LinearRegression())
# ])
# # Helps to train the model 
# pipeline.fit(X_train, y_train)
# y_pred_scaled = pipeline.predict(X_test)
# r2_scaled = r2_score(y_test, y_pred_scaled)
# print(f"Scaled Linear Regression R²: {r2_scaled:.4f}")

import seaborn as sns
graph=sns.displot(reg_pred-y_test,kind='kde')
print(graph)
plt.show()




