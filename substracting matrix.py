import numpy as np
# Create matrix
matrix_a = np.array([[1, 1, 1],
 [1, 1, 1],
 [1, 1, 2]])
# Create matrix
matrix_b = np.array([[1, 3, 1],
 [1, 3, 1],
 [1, 3, 8]])
# Add two matrices
np.add(matrix_a, matrix_b)
np.array([[ 2, 4, 2],
 [ 2, 4, 2],
 [ 2, 4, 10]])
# Subtract two matrices
print(np.subtract(matrix_a, matrix_b))
