# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()  # create an object of the LinearRegression class
lin_reg.fit(x, y)  # training the model using fit() method

# Training the Polynomial Regression model on the whole dataset

# two steps:
# step 01.create the metric of powered features (x1, x1^2, x1^3..)
from sklearn.preprocessing import PolynomialFeatures

# create an object of this class to create the metric of the features
# degree=2 ( y = b0 + b1x1 + b2x1^2
poly_reg = PolynomialFeatures(degree=4)

# transform the metrics of single features into the new metrics of features (transform the input features into
# polynomial terms) with x1 and x1^2
x_poly = poly_reg.fit_transform(x)

# step 02: build a linear regression model using new features
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color='red')
# need to use the transformed features.
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))