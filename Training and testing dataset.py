from sklearn.model_selection import train_test_split
import numpy as np
# Example dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) # Features
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1]) # Labels
# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Features:\n", X_train)
print("Testing Features:\n", X_test)


  '''output:
Training Features:
 [[ 6]
 [ 1]
 [ 8]
 [ 3]
 [10]
 [ 5]
 [ 4]
 [ 7]]
Testing Features:
 [[9]
 [2]]'''
