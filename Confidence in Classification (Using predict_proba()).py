from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Load dataset
data = load_iris()
X, y = data.data, data.target
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Predict probabilities
probs = model.predict_proba(X_test)
# Get the predicted classes and their confidence
predicted_classes = np.argmax(probs, axis=1)
confidence_scores = np.max(probs, axis=1)
# Print example results
for i in range(5):
    print(f"Instance {i+1}: Predicted class = {predicted_classes[i]}, Confidence ={confidence_scores[i]:.2f}")


    '''output:
Instance 1: Predicted class = 1, Confidence =0.97
Instance 2: Predicted class = 0, Confidence =1.00
Instance 3: Predicted class = 2, Confidence =0.98
Instance 4: Predicted class = 1, Confidence =0.99
Instance 5: Predicted class = 1, Confidence =0.92
'''
