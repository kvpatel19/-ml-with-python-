import pandas as pd
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('D:\\kriyanshi\\bcaresult1.csv')
data
lbl=LabelEncoder()
data["name"]=lbl.fit_transform(data["name"])
data
