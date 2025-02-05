from sklearn.linear_model import LinearRegression
import numpy as np
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([2,4,5,4,5])
model=LinearRegression()
model.fit(x,y)
print("slope(m):",model.coef_[0])
print("intercept(c):",model.intercept_)
prediction=model.predict(x)
print("prediction:",prediction)

'''output
slope(m): 0.6
intercept(c): 2.2
prediction: [2.8 3.4 4.  4.6 5.2]

'''
