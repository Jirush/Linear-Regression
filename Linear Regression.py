# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 08:16:26 2018

@author: Dell
"""

import pandas as pd
import matplotlib as plt

dataset = pd.read_csv('Salary_data.csv')
x = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

x_train = pd.DataFrame(x_train)
x_train = x_train.reshape(-1,1)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)