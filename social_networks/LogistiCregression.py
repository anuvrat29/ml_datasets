# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:35:21 2018

@author: testuser

Logistic Regression Example
Dataset: Social_Network_Ads.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

features = dataset.iloc[:,[2,3]].values
label = dataset.iloc[:,[4]].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

y_prediction = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_prediction)

np.min(features[:,0]) #18
np.max(features[:,0]) #60

np.min(features[:,1]) #15000
np.max(features[:,1]) #150000