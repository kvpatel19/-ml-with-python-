from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labels=['cat','dog','mouse']
data=pd.get_dummies(labels)
scaler=RobustScaler()
robust_scaled_data=scaler.fit_transform(data)
print(robust_scaled_data)


'''output
[[2. 0. 0.]
 [0. 2. 0.]
 [0. 0. 2.]]'''
