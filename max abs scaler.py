import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
d=pd.read_csv("D:\\kriyanshi\\bcaresult.csv")
data=pd.get_dummies(d["name"])
scaler=MaxAbsScaler()
maxabs_scaled_data=scaler.fit_transform(data)
print(maxabs_scaled_data)


'''output
[[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MaxAbsScaler
d=pd.read_csv("D:\\kriyanshi\\bcaresult.csv")
data=pd.get_dummies(d["name"])
scaler=MaxAbsScaler()
maxabs_scaled_data=scaler.fit_transform(data)
print(maxabs_scaled_data)
for i in d.select_dtypes(include="number").columns:
    sb.histplot(data=d,x=i)
plt.show()

