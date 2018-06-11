# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:44:43 2018

@author: Ashwin Dhakaita
"""

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score

data = pd.read_csv("H://train_titanic.csv")
test = pd.read_csv("H://test_titanic.csv")

data.drop('PassengerId',axis=1 , inplace = True)
data.drop('Name',axis=1 , inplace = True)
data.drop('Ticket',axis=1,inplace = True)

Id = test['PassengerId']
test.drop('PassengerId',axis=1,inplace=True)
test.drop('Name',axis=1 , inplace = True)
test.drop('Ticket',axis=1,inplace = True)

for i in range(891):
    if data.iloc[i,1]==1:
        data.iloc[i,7] = 'U'
    else:
        data.iloc[i,7] = 'L'
        
for i in range(test.shape[0]):
    if test.iloc[i,0]==1:
        test.iloc[i,6] = 'U'
    else:
        test.iloc[i,6] = 'L'

imp = Imputer(missing_values=np.NaN,strategy="median",axis=0)
data['Age'] = imp.fit_transform(data[['Age']]).ravel()
test['Age'] = imp.transform(test[['Age']]).ravel()

imp = Imputer(missing_values=np.NaN,strategy="median",axis=0)
data['Fare'] = imp.fit_transform(data[['Fare']]).ravel()
test['Fare'] = imp.transform(test[['Fare']]).ravel()

data.iloc[61,8] = 'S'
data.iloc[829,8] = 'S'

lbd = LabelEncoder()
data['Cabin'] = lbd.fit_transform(data['Cabin'])
test['Cabin'] = lbd.transform(test['Cabin'])

lbd2 = LabelEncoder()
data['Embarked'] = lbd2.fit_transform(data['Embarked'])
test['Embarked'] = lbd2.transform(test['Embarked'])

lbd3 = LabelEncoder()
data['Sex'] = lbd3.fit_transform(data['Sex'])
test['Sex'] = lbd3.transform(test['Sex'])
'''
lbd4 = LabelEncoder()
data['Ticket'] = lbd4.fit_transform(data['Ticket'])

lbd5 = LabelEncoder()
data['Name'] = lbd5.fit_transform(data['Name'])
'''
#data['SibSp'] = data['SibSp'] + data['Parch']
#test['SibSp'] = test['SibSp'] + test['Parch']
#data.drop('Parch',axis=1,inplace=True)
#test.drop('Parch',axis=1,inplace=True)

for x in ['Age','SibSp','Parch','Fare']:
    StndSc = StandardScaler()
    data[x] = StndSc.fit_transform(data[x].values.reshape(891,1))
    test[x] = StndSc.transform(test[x].values.reshape(test.shape[0],1))

X = data.iloc[:,1:]
y = data.iloc[:,0]

#trainX , testX , trainy , testy = train_test_split(X,y,test_size=.33)    
gdb = GradientBoostingClassifier(n_estimators=250,max_depth=3)
gdb.fit(X,y)

pred = gdb.predict(test)
#print(accuracy_score(list(testy.values),pred))

scores = cross_val_score(gdb,X,y,cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
df = pd.DataFrame(index = ['PassengerId','Survived'])

df = pd.DataFrame({'PassengerId':Id , 'Survived':pred})
df.to_csv("H://Submissions_Titanic.csv",index=False)
print(df.iloc[:,:])
'''
df['PassengerId'] = Id
df['Survived'] = pred
'''
