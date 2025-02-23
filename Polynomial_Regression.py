import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

# Here a matrix of 100 rows and 1 column withc range 0 to 1 and 6* shows 0 to 6 and -3 shows range between -3 to 3 
X = 6 * np.random.rand(100, 1) - 3

# quadratic equation used- y=0.5x^2+1.5x+2+outliers
y =0.5 * X**2 + 1.5*X + 2 + np.random.randn(100, 1)
print(X)
print(y)
plt.scatter(X,y,color='g')
plt.xlabel('X dataset')
plt.ylabel('Y dataset')
plt.show()
 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize the Linear Regression model
regression_1=LinearRegression() 
model=regression_1.fit(X_train,y_train)
print(model)
y_pred =regression_1.predict(X_test)
score=r2_score(y_test,y_pred)
print(f"Linear Regression R²: {score:.4f}")

plt.plot(X_train,regression_1.predict(X_train),color='r')
plt.scatter(X_train,y_train)
plt.xlabel("X Dataset")
plt.ylabel("Y")
plt.show()

# Initialize the Polynomial Features model

# Creates polynomial features up to the 2nd degree (quadratic)
# Includes an extra column of ones (for the intercept term in regression)
poly=PolynomialFeatures(degree=2,include_bias=True)

# Transforms X_train into a new feature matrix that includes polynomial terms
X_train_poly=poly.fit_transform(X_train)

# Applies the same transformation learned from X_train to X_test (without learning again).
# Ensures that the test set uses the same polynomial expansion.
X_test_poly=poly.transform(X_test)
print(X_train_poly)
print(X_test_poly)

# Again Initialize the Linear Regression model
regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score=r2_score(y_test,y_pred)
print(f" New Linear Regression R²: {score:.4f}")
print(regression.coef_)
print(regression.intercept_)
plt.scatter(X_train,regression.predict(X_train_poly), color="blue", label="Predicted" )
plt.scatter(X_train,y_train ,color="Red",label="Actual")
plt.legend()
plt.show()
# poly=PolynomialFeatures(degree=3,include_bias=True)
# X_train_poly=poly.fit_transform(X_train)
# X_test_poly=poly.transform(X_test)
# X_train_poly
# from sklearn.metrics import r2_score
# regression = LinearRegression()
# regression.fit(X_train_poly, y_train)
# y_pred = regression.predict(X_test_poly)
# score=r2_score(y_test,y_pred)
# print(score)


##### Polynomial Regression to avoid overfitting 

# X represent 1 to 10
# Reshape provides a 2D array  10 to 1
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# Y shows Quadratic relationship with x values
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Converts the original X values into a polynomial feature space (degree = 2)
# This means instead of just X, the model will consider X and X² terms
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Apply Ridge regression (regularization) to avoid overfitting
# Regularization parameter (higher alpha = more regularization)
model = Ridge(alpha=1.0)  
model.fit(X_poly_train, y_train)

y_pred = model.predict(X_poly_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Plotting the graph
plt.scatter(X_test, y_test, color='blue', label='Actual Data')  
plt.scatter(X_test, y_pred, color='red', label='Predicted Data')  

# Create a smooth regression line by predicting over a range of values
X_range = np.linspace(min(X_test), max(X_test), 100).reshape(-1, 1)  # Create smooth range for X
X_range_poly = poly.transform(X_range)  # Transform the range using the polynomial features
y_range_pred = model.predict(X_range_poly)  # Get predictions for the smooth range

plt.plot(X_range, y_range_pred, color='green', label='Regression Line')  # Green curve for polynomial regression

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Regularization (Ridge)')
plt.legend()
plt.show()