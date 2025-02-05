import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
d=pd.read_csv("D:\\kriyanshi\\bcaresult.csv")
d
for i in d.select_dtypes(include="number").columns:
    sb.histplot(data=d,x=i)
plt.show()
