import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
n_redundant=0, random_state=42)
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train a linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
# Make predictions
y_pred_linear = linear_svm.predict(X_test)
# Evaluate performance
print("Linear SVM:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_linear):.2f}")
print(classification_report(y_test, y_pred_linear))
# Plot decision boundary
plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z = linear_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.3, colors=['blue', 'red'])
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
plt.title("Linear SVM Decision Boundary")
plt.show()


'''output:'''
