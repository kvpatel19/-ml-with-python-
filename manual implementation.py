import numpy as np
data=np.array([[2.0,3.0],[4.0,5.0],[6.0,7.0]])
mean_removed=data-np.mean(data,axis=0)
print(mean_removed)
