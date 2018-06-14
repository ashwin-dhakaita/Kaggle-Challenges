# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:13:44 2018

@author: Ashwin Dhakaita
"""
#import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

#function to assign labels to a new variable based on months in which more fires occur
def month_imp(x):
    if x in [1,11]:
        return 3
    elif x in [3,5,7]:
        return 2
    elif x in [2,6,0,10]:
        return 1
    else:
        return 0

#reading the data file
data = pd.read_csv("H://forestfires.csv")

#label the ordinal variable to check correlation
lbd = LabelEncoder()
data['month'] = lbd.fit_transform(data['month'])
data['day'] = lbd.fit_transform(data['day'])

#create a correlation map using seaborn with matplotlib
plt.figure(figsize=(15,6))
sns.heatmap(data.corr(),annot=True)

#create a factorplot to detect outliers
sns.factorplot(x='FFMC',y='ISI',data=data)

#drop outliers
data.drop(data.index[data['ISI']>20],inplace=True)

#create a distribution plot againt area
sns.distplot(data['area'])
plt.legend(col = 'best')

#drop outliers
data.drop(data.index[data['area']>250],axis=0,inplace=True)

#transform the area vector by taking its log to decrease skewness
data['area'] = data['area'].map(lambda x: np.log(x+1) if x>0 else 0)

#perform scaling over numerical predictors
l = ['X','Y','FFMC','DC','DMC','ISI','temp','RH','wind','rain']
for i in l:
    StndSc = StandardScaler()
    data[i] = StndSc.fit_transform(data[i].values.reshape(data.shape[0],1))

#create a new numerical predictor DDMC 
data['DDMC'] = data['DMC']/data['DC']

#create a new ordinal variable based on months and their associated fire intensities 
data['Month_Imp'] = data['month'].map(month_imp)

y = data['area']
data.drop(['area'],axis=1,inplace=True)
X = data

#performing train and test split
trainX , testX , trainy , testy = train_test_split(X,y,test_size=.4)

regressors = [LinearRegression , DecisionTreeRegressor , RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor , BaggingRegressor]
rmse = []

#calculate root mean squared error of various regressors
for reg in regressors:
    R = reg()
    R.fit(trainX,trainy)
    pred = R.predict(testX)
    rmse.append(mean_squared_error(testy.values , pred))
    
df = pd.DataFrame({'Regressors':['LinearRegression','DecisionTree','RandomForest','AdaBoost','GradientBoosting','Bagging'],'RMSE':rmse})

#creating a final bar plot against their corresponding performances
sns.barplot(x = 'RMSE' , y='Regressors' ,data = df, orient='h')