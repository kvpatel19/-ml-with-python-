import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict with confidence intervals
y_pred = model.predict(X_test)
residuals = y_train - model.predict(X_train)
std_dev = np.std(residuals)
# Compute confidence intervals
confidence = 1.96 * std_dev # For ~95% confidence
lower_bounds = y_pred - confidence
upper_bounds = y_pred + confidence
# Print example results
for i in range(5):
    print(f"Prediction: {y_pred[i]:.2f}, 95% CI: [{lower_bounds[i]:.2f},{upper_bounds[i]:.2f}]")

    '''output:
Prediction: -59.25, 95% CI: [-75.70,-42.80]
Prediction: 65.16, 95% CI: [48.71,81.61]
Prediction: 35.66, 95% CI: [19.21,52.11]
Prediction: -17.75, 95% CI: [-34.20,-1.29]
Prediction: -10.74, 95% CI: [-27.20,5.71]
'''
