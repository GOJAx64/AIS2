import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import r2_score 

np.random.seed(42) 

m = 100 
x = 6 * np.random.rand(m, 1) - 3 
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1) 

poly = PolynomialFeatures( degree=2, include_bias=False ) 
lin_reg = LinearRegression()
std_scaler = StandardScaler()

model = Pipeline([('poly_features', poly      ), 
                  ('scaler',        std_scaler), 
                  ('lin_reg',       lin_reg   )]) 

model.fit(x, y)
y_pred = model.predict(x)
x_new = np.linspace(-3, 3, 100).reshape(100, 1)
y_new = model.predict(x_new)

plt.figure()

plt.plot( x, y, 'b.' )
plt.xlabel( 'Ox1$', fontsize=18 )
plt.ylabel( '$y$', fontsize=18 )
plt.axis( [-3, 3, 0, 10] )
plt.plot( x_new, y_new, 'r-', linewidth=2, label='predictions' ) 
plt.legend( loc='upper left', fontsize=18 ) 

print(r2_score(y, y_pred)) 
