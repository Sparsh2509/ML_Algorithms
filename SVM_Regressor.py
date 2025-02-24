from sklearn.svm import SVR

# Generate a random regression problem
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# N samples means no of data points 
# N feature to generate feature of regression problem here 1D
# Noise Adds slight randomness to make the dataset more realistic
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Regressor with kernal rbf Radial Basis Function to capture non linear relationship 
svm_regressor = SVR(kernel='rbf')

model =svm_regressor.fit(X_train, y_train)
print(model)

y_pred = svm_regressor.predict(X_test)
score =svm_regressor.score(X_test, y_test)
print(f"R-squared Score: {score:.4f}")


# Hyperparameter tuning using GridSearchCV
# C show regularization strength
# Gamma is coeff of kernel which influence  of the far points in single training example reaches

parameter = { 'C': [0.1, 1, 10, 100, 1000],'gamma':  [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}


grid_search = GridSearchCV(SVR(), param_grid=parameter, cv=5)

grid_search.fit(X_train, y_train)
print(grid_search)

print("Best Hyperparameters:", grid_search.best_params_)

# Plotting the results
plt.scatter(X_test, y_test, color='red', label="Actual Values")
plt.scatter(X_test, y_pred, color='blue', label="Predicted Values")
plt.title('SVM Regressor Results')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()