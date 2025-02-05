from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
labels=['cat','dog','mouse']
data=pd.get_dummies(labels)
scaler= StandardScaler()
standardized_data=scaler.fit_transform(data)
print(standardized_data)


   ''' output
[[ 1.41421356 -0.70710678 -0.70710678]
 [-0.70710678  1.41421356 -0.70710678]
 [-0.70710678 -0.70710678  1.41421356]]'''
