# importin ridge regression by sklearn.linear model
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ridge_regressor=Ridge()
# ridge_regressor
X=dataset
y=df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
parameters={'alpha':[1,2,5,10,20,30,40,50,60,70,80,90]}
ridgecv=GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(X_train,y_train)
print(ridgecv.best_params_)
print(ridgecv.best_score_)
ridge_pred=ridgecv.predict(X_test)
import seaborn as sns
sns.displot(ridge_pred-y_test,kind='kde')
score=r2_score(ridge_pred,y_test)
score
