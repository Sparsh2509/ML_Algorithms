import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report , accuracy_score

iris = load_iris()
X = iris.data 
y = iris.target  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor with specific parameters
# Mean Squared Error: 0.0083
# R² Score: 0.9881

regressortree = DecisionTreeRegressor(
    # criterion="squared_error",
    # max_depth=4,  
    # min_samples_split=3,
    # min_samples_leaf=2,
    # random_state=42
)

##### Using grid search cv for more generalized algo

param_grid = {
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'max_depth': [3, 4, 5, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4]
}

# Mean Squared Error: 0.0086
# R² Score: 0.9876
# Accuracy: 0.9667

grid_search = GridSearchCV(regressortree, param_grid, cv=5, scoring='r2',)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

# Convert Regression Output to Class Labels (For Classification Report)
y_pred_class = np.round(y_pred).astype(int)  # Round to nearest class



mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_class)

class_report =classification_report(y_test, y_pred_class, target_names=iris.target_names)



print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)


plt.figure(figsize=(12, 6))
plot_tree(best_model, feature_names=iris.feature_names, filled=True, rounded=True, class_names=iris.target_names)
plt.title("Optimized Decision Tree Regression - Iris Dataset")
plt.show()

# Without tuning 

# plt.figure(figsize=(12, 6))
# plot_tree(regressortree, feature_names=iris.feature_names, filled=True, rounded=True, class_names=iris.target_names)
# plt.title("Decision Tree Regression - Iris Dataset")
# plt.show()