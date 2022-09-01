# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 07:15:33 2022

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

np.random.seed(42)
m = 300
r = 0.5
ruido= r *np.random.randn(m, 1)
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + ruido

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

from sklearn.kernel_ridge import KernelRidge

model = KernelRidge(alpha=0.1, kernel='rbf')
model.fit(xtrain, ytrain)

print('Train: ',model.score(xtrain, ytrain))
print('Test: ',model.score(xtest, ytest))

xnew = np.linspace(-3, 3, 50).reshape(-1, 1)
ynew= model.predict(xnew)

plt.plot(xnew,ynew, '-k', linewidth=3)

plt.plot(xtrain, ytrain, 'b.')
plt.plot(xtest,  ytest,  'r.')
plt.xlabel('$x$', fontsize=18 )
plt.xlabel('$y$', fontsize=18 )
plt.axis([-3, 3, 0, 10])
plt.show()

#%matplolib auto /%matplolib  inline
'''
 aplha: mas grande tiende a subentrenarse, mas pequeño sobreentrena
'''