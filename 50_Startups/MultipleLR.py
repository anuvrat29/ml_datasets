# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:14:09 2018

@author: testuser

Multiple Linear Regression

"""
#Step1: Import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step2: Import dataset
dataset = pd.read_csv('50_Startups.csv')

#Step3: Get the features and label
features = dataset.iloc[:,[0,1,2,3]].values
label = dataset.iloc[:,[4]].values

#Step4: Label and One Hot Encoding for state field
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
features[:,3]=encoder.fit_transform(features[:,3])

ohe=OneHotEncoder(categorical_features=[3])
features=ohe.fit_transform(features).toarray()

#Situation: Dummy Variable Trap ---> Dummy Variable > 3
#SKLearn is already adapt to handle dummy variable trap error
#To handle the situation of dummy variables in manual mode(not using sklearn)
#Delete the first dummy field.

#Seperate the data with train and test splits

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
                                               
                                  features,
                                  label,
                                  test_size=0.25,
                                  random_state=0             
                                               )

#Create our model
#Multiple Linear Regression Model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

#Test our model

y_predictions = model.predict(X_test)

#Get R2 score

from sklearn.metrics import r2_score
r2_score(y_test,y_predictions)





