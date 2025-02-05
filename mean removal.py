import pandas as pd
from sklearn import preprocessing
d=pd.read_csv("D:\\kriyanshi\\bcaresult.csv")
d
ohe=pd.get_dummies(d["name"])
ohe
stdscale=preprocessing.scale(ohe)
print("mean data:",stdscale.mean(axis=0))
print("standard mean data:",stdscale.std(axis=0))
