from sklearn.preprocessing import StandardScaler
import numpy as np
data=np.array([[10,2],[20,4],[30,6]])
scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)
print(scaled_data)
