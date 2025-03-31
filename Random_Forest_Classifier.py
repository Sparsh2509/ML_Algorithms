import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a Random Forest model

#Score 1.000
rf_clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Print Accuracy Score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Extract and visualize a single decision tree from the Random Forest
plt.figure(figsize=(12, 8))
plot_tree(rf_clf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("A Single Tree from Random Forest - Iris Dataset")
plt.show()