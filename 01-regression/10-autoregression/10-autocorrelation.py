import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('daily-min-temperatures.csv')

#x = np.asanyarray(df[['Temp']])

'''
plt.plot(x)
#p = 365
plt.scatter(x[1:], x[:-1])
print(np.corrcoef(x[p:].T, x[:-p].T))

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Temp)
'''

data = pd.DataFrame(df.Temp)
p = 5

for i in range(1, p+1):
    data = pd.concat([data, df.Temp.shift(-i)], axis = 1)
    
data = data[:-p]

x = np.asanyarray(data.iloc[:,:-1])#:-1 y 0
y = np.asanyarray(data.iloc[:,-1]) #-1



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

from sklearn.svm import SVR

model = SVR(gamma='scale', C=1.6, epsilon=0.2, kernel='rbf')
model.fit(xtrain, ytrain)

print('Train: ',model.score(xtrain, ytrain))
print('Test: ',model.score(xtest, ytest))

'''
xnew = np.linspace(-3, 3, 50).reshape(-1, 1)
ynew= model.predict(xnew)
'''