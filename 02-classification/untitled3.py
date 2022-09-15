import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()

x = iris['data'][:, (2,3)]
y = iris['target']

plt.plot(x[y==0,0], x[y==0, 1], 'g^', label='Iris-Setosa')
plt.plot(x[y==1,0], x[y==1, 1], 'bs', label='Iris-Versicolor')
plt.plot(x[y==2,0], x[y==2, 1], 'yo', label='Iris-Virginica')

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

softmax_reg = LogisticRegression(multi_class='multinomial')
softmax_reg.fit(xtrain, ytrain)

print( 'Train: ', softmax_reg.score(xtrain, ytrain) )
print( 'Test: ' , softmax_reg.score(xtest, ytest) )

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

ypred = softmax_reg.predict(xtest)
print('Confusion matrix:')
print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))