from sklearn.preprocessing import Binarizer
data=[[1.5,-3.0,2.1],[0.1,3.4,-1.2]]
binarizer=Binarizer(threshold=1.0)
data=binarizer.fit_transform(data)
print(data)
binarizer=Binarizer(threshold=15)
binary_data=binarizer.fit_transform(data)
print(binary_data)


'''output
[[1. 0. 1.]
 [0. 1. 0.]]
[[0. 0. 0.]
 [0. 0. 0.]]'''
