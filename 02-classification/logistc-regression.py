import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv('diabetes.csv')
'''
Otra forma de ver los datos, es usar estas graficas
import seaborn as sns


corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
'''

x = np.asanyarray(data.drop(columns=['Outcome']))
y = np.asanyarray(data[['Outcome']]).ravel()

xtrain, xtest, ytrain, ytest = train_test_split(x, y)
model = Pipeline([
                 ('scaler', StandardScaler()),
                 ('logit', LogisticRegression())
                ])

model.fit(xtrain, ytrain)

print( 'Train: ', model.score(xtrain, ytrain) )
print( 'Test: ' , model.score(xtest, ytest) )

coeff = list( np.abs(model.named_steps['logit'].coef_[0]) )
coeff = coeff / np.sum(coeff)
labels = list(data.drop(columns=['Outcome']).columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values( by=['importance'], ascending=True, inplace=True )
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh')
plt.xlabel('Importance')