from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures



np.random.seed(42)
m = 200
x = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * x + np.random.randn(m, 1) / 0.5
xnew = np.linspace(0, 3, 100).reshape(100, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

model = Pipeline([('poly', PolynomialFeatures(degree=20, include_bias=False)), 
                  ('scaler', StandardScaler()), 
                  ('reg_lin', Lasso(alpha=0.1))])#Lasso, Ridge, ElasticNet

model.fit(xtrain, ytrain)

print('Train R2: ', model.score(xtrain, ytrain))
print('Test R2: ', model.score(xtest, ytest))

plt.plot(xtrain, ytrain, 'b.')
plt.plot(xtest,  ytest,  'r.')
plt.plot(xnew, model.predict(xnew), '-k')

