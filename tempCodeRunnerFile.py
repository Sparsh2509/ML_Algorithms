# Plot decision boundary
# def plot_decision_boundary(X, y, model):
#     h = 0.02  # Step size in mesh
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
    
#     plt.contourf(xx, yy, Z, alpha=0.3)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title("Optimized SVM Decision Boundary")
#     plt.show()

# plot_decision_boundary(X_train, y_train, best_svm)
