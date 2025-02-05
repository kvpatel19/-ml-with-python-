from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# Load dataset
data = load_iris()
X, y = data.data, data.target
# Initialize model
model = RandomForestClassifier()
# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# Print results
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")


'''output:
Cross-validation scores: [0.96666667 0.96666667 0.9        0.9        1.        ]
Mean accuracy: 0.9467
Standard deviation: 0.0400
'''
