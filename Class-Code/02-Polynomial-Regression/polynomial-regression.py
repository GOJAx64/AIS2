import numpy as np 
import numpy.random as rnd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score  

np.random.seed(42) 

m = 100 
x = 6 * np.random.rand(m, 1) - 3 
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)


plt.plot(x, y, 'b.') 
plt.xlabel('$x_1$', fontsize=18) 
plt.ylabel('$y$', fontsize=18, rotation=0) 
plt.axis([-3, 3, 0, 10]) 

poly = PolynomialFeatures( degree=2, include_bias=False ) 
x_poly = poly.fit_transform(x)

lin_reg = LinearRegression() 
lin_reg.fit(x_poly, y) 

x_new = np.linspace(-3, 3, 100).reshape(100, 1) 
x_new_poly = poly.fit_transform(x_new) 
y_new = lin_reg.predict(x_new_poly) 
plt.plot(x_new, y_new, 'r-', linewidth=2, label='predictions')
plt.legend( loc='upper left', fontsize=18) 

print(r2_score, lin_reg.predict(x_poly)) 