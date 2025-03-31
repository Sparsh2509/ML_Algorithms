import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [10, 50, 100],  # Number of trees in the forest
    "max_depth": [3, 5, None],      # Depth of trees
    "min_samples_split": [2, 5, 10], # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 4],   # Minimum samples at leaf node
    "criterion": ["gini", "entropy"] # Splitting criteria
}

# Train a Random Forest model

#Score 1.000
# random_classifier = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
# random_classifier.fit(X_train, y_train)

# Score 1.000
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Make predictions
y_pred = best_model.predict(X_test)

# Print Accuracy Score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Extract and visualize a single decision tree from the Random Forest
plt.figure(figsize=(12, 8))
plot_tree(best_model.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("A Single Tree from Random Forest - Iris Dataset")
plt.show()