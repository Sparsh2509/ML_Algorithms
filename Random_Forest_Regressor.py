import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (continuous for regression)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor

# Mean Squared Error (MSE): 0.0021
# R² Score: 0.9970
# rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
# rf_regressor.fit(X_train, y_train)

param_grid = {
    'n_estimators': [50, 100],       # Reduced number of trees
    'max_depth': [2,4,6],             # Limit the depth of the trees
    'min_samples_split': [2, 5],      # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2],       # Minimum samples required to be at a leaf node
    'bootstrap': [True, False],       # Bootstrap sampling
    'criterion': ['absolute_error', 'squared_error', 'poisson', 'friedman_mse']      # Criterion for quality of a split (mean squared error or mean absolute error)
}

# Best parameters found:  {'bootstrap': True, 'criterion': 'squared_error', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
# Mean Squared Error (MSE): 0.0014
# R² Score: 0.9980

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Make predictions
# y_pred = rf_regressor.predict(X_test)

best_rf = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

y_pred = best_rf.predict(X_test)

# Print evaluation metrics only once
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Extract and visualize a single decision tree from the Random Forest
plt.figure(figsize=(12, 8))
plot_tree(best_rf.estimators_[0], feature_names=iris.feature_names, filled=True, rounded=True)
plt.title("A Single Tree from Random Forest Regressor - Iris Dataset")
plt.show()


# plt.scatter(y_test, y_pred, color='blue')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted - Random Forest Regressor")
# plt.show()