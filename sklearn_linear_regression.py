# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:00:52 2025

@author: laket
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error  

# Read in excel files
df = pd.read_excel("CoviddataML1.xlsx", usecols=['date']) # Import columns by name
df_1 = pd.read_excel("CoviddataML1.xlsx", usecols=['avg.deaths'])

# data sanatization if necessary
#df.fillna(method ='ffill', inplace = True)

# Create numpy arrays and flatten them if they are multi-dimensional
X = np.array(df).reshape(-1, 1) 
y = np.array(df_1).reshape(-1, 1) 

#print(X)
#print(y)

# Plot the data 
plt.title("scatter of data")
plt.scatter(X, y, color = 'r')
plt.show()

#plt.title("plotting data")
#plt.show()

#fitting the data
model = LinearRegression()
model.fit(X, y)
plt.scatter(X,y, color = 'r')
line = model.coef_[0][0]*X + model.intercept_[0]
plt.plot(X, line, color = 'r')
print(f'Equation of the data line: y={model.coef_[0][0]}x + {model.intercept_[0]}')

# Splitting the data into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25) 

model.fit(X_train, y_train)
line = model.coef_[0][0]*X + model.intercept_[0]
plt.plot(X, line, color = 'b')
print(f'Equation of the training line: y={model.coef_[0][0]}x + {model.intercept_[0]}')
plt.title("Compare line fit between test (blue) and real data(red)")
plt.show()

print("Model R Squared score: " + str(model.score(X_test, y_test))) 

y_pred = model.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k') 

# Show Data scatter of predicted values 
plt.show() 

#Evaluate model performance
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred) 
#squared True returns MSE value, False returns RMSE value. 
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True 
rmse = root_mean_squared_error(y_true=y_test,y_pred=y_pred) 

print("MAE:",mae) 
print("MSE:",mse) 
print("RMSE:",rmse)

