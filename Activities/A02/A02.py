#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 18:53:17 2022

@author: alberto
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model 

data = pd.read_csv('countries.csv')
data_mex = data[data.country == 'Mexico']

x = np.asarray(data_mex[['year']]) 
y = np.asarray(data_mex[['lifeExp']]) 

model = linear_model.LinearRegression() 
model.fit(x, y) 
y_pred = model.predict(x) 

plt.figure() 
plt.scatter(x, y) 
plt.plot(x, y_pred, '--r') 