import numpy as np
from scipy import sparse
matrix=np.array([[0,0],[0,1],[3,0]])
print("simple matrix")
print(matrix)
matrix_sparse=sparse.csr_matrix(matrix)
print("sparse matrix")
print(matrix_sparse)
