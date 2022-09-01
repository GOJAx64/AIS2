import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#from sklearn import r2_score

data = pd.read_csv('home_data.csv')

# price como variable y
y = np.asanyarray(data[('price')])

#eliminar la fecha y id
# el resto de las variables para ver cual es la mas importante
x = np.asanyarray(data.drop(['id', 'date', 'price'], axis=1))

scaler = StandardScaler()
x = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

#Entrenar regresor lineal
#model =  linear_model.LinearRegression()
model = Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)), 
                  ('scaler', StandardScaler()), 
                  ('reg_lin', Ridge(alpha=-1e-05))])#0.1

model.fit(x, y)

print('Train R2: ', model.score(xtrain, ytrain))
print('Test R2: ', model.score(xtest, ytest))

#Sacar coeficientes
'''
coef = model.coef_

#Ordenar los coheficientes segun sumagnitud
df = pd.DataFrame(columns=['feauture', 'coef'])

columns = (data.drop(["id", 'date', 'price'], axis=1)).columns

df['feauture'] = columns
df['coef'] = abs(coef)

df = df.sort_values(by='coef')
#Reportar el nombre de las variables ( de la mas  a le menos importante )
'''

