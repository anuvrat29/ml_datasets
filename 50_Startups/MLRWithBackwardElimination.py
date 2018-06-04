# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:11:25 2018

@author: testuser

Multiple Linear Regression with Backward Elimination!
"""
import numpy as np
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

features = dataset.iloc[:,[0,1,2,3]].values
label = dataset.iloc[:,[4]].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode = LabelEncoder()
features[:,3]=encode.fit_transform(features[:,3])
ohe = OneHotEncoder(categorical_features=[3])
features = ohe.fit_transform(features).toarray()

#COMMON ERROR OR COMMON PROBLEM THAT THE MODEL WILL FACE
#DUMMY VARIABLE TRAP
#Usual solution to get rid of DUMMY VARIABLE TRAP is to 
#delete one dummy column
#Note: If you are using sklearn, it will automatically handle
#Dummy Variable trap issue!!!




from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=0.25,random_state=0)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
#######

#Lets perform Backward Elimination technique to figure out
#the best features that influence the label positively

#Step1: Select the Significance Level
#Its all about understanding which featyure has more
#statistical relationshop with the label.
#Value of SL range from 0 to 1
# default suggested value to start with is 0.05 !!!

#Step2: Perform ALL-IN (b0,b1,b2,b3,b4,...bn features!!)
features = np.append(arr=np.ones((50,1)).astype(int) , values= features, axis = 1)
X_opt = features[:,[0,1,2,3,4,5,6]]

#Step3: Consider the feature with the highest p-value
# if (p > sl) Go to Step4 else YOUR MODEL IS READY !!!
#To identify the P-value, we will use OLS formula

import statsmodels.formula.api as sm
#Iteration1: highest p-value is 0.608 for x5


#OLS(Feature,label)

model_OLS = sm.OLS(endog=label, exog=X_opt).fit()
model_OLS.summary()

#Iteration2

X_opt = features[:,[0,1,2,3,4,6]]
model_OLS = sm.OLS(endog=label, exog=X_opt).fit()
model_OLS.summary()

#Iteration 3
X_opt = features[:,[0,1,2,3,4]]
model_OLS = sm.OLS(endog=label, exog=X_opt).fit()
model_OLS.summary()











