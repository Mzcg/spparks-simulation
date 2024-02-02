#the results are not reliable 20240201/4:10pm, do not use now until fix.
#we decide not use the approximate regression value, but use exact valuesu code in command_line_arg.sh using "case"

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Your data
speed = np.array([3, 9, 15, 21, 27, 33, 39, 45])
frame = np.array([4500, 1500, 900, 650, 500, 450, 350, 300])

# Reshape the input array
speed_reshaped = speed.reshape(-1, 1)

# Polynomial regression with degree 2
poly_features = PolynomialFeatures(degree=4)
speed_poly = poly_features.fit_transform(speed_reshaped)

# Fit the model
model = LinearRegression()
model.fit(speed_poly, frame)

# Get the coefficients
coefficients = model.coef_
intercept = model.intercept_

# Display the equation
equation = f"Frame = {intercept:.2f}"
for i in range(len(coefficients) - 1, 0, -1):
    equation += f" + {coefficients[i]:.2f} * Speed^{i}"

print("Polynomial Equation:", equation)

# Predict for a new value
new_speed = np.array([[3]])
new_speed_poly = poly_features.transform(new_speed)
predicted_frame = model.predict(new_speed_poly)

# Visualization
x_values = np.linspace(min(speed), max(speed), 100).reshape(-1, 1)
x_values_poly = poly_features.transform(x_values)
y_values = model.predict(x_values_poly)

plt.scatter(speed, frame, label='Data Points')
plt.plot(x_values, y_values, color='red', label='Polynomial Regression')
plt.xlabel('Speed')
plt.ylabel('Frame')
plt.legend()
plt.show()

