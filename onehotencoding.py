from pandas import*
data=read_csv("D:\\kriyanshi\\bcaresult.csv")
data
xdata=get_dummies(data["name"])
xdata
