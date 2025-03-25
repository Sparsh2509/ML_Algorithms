import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use first two features for visualization
y = iris.target

print(X)
print(y)
# print(iris.target_name)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 100],        # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'gamma': ['scale', 'auto', 0.1, 0.5]  # Kernel coefficient
}

# Perform GridSearchCV with cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(grid_search)

# Best hyperparameters
print(f"Best parameters: {grid_search.best_params_}")

# Train SVM model with best parameters
best_svm = grid_search.best_estimator_

print(best_svm)

# Predictions
y_pred = best_svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
ax = plt.gca()

# Create grid points to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))

# Predict for each point in the grid
Z = grid_search.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

# Plot the training points
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,  edgecolors='black', marker='o', s=80, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Classifier with Decision Boundary')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()


