from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
x=np.array([[1],[2],[3],[4],[5],[6]])
y=np.array([0,0,0,1,1,1])
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print("accuracy:",accuracy)
print("confusion matrix:\n",conf_matrix)
print("classification report: \n",report)
