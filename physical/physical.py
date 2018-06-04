# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Physical.csv")
features = dataset.iloc[:,1:].values
label = dataset.iloc[:,0].values

#Training and testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state=9)

#Create a model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#Predictions
y_pred = model.predict(x_test)

#Accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
