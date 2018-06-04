# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Advertising.csv")
features = dataset.iloc[:,1:4].values
label = dataset.iloc[:,4].values

#Handle Missing data
from sklearn.preprocessing import Imputer
imputerNaN = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
features = imputerNaN.fit_transform(features[:,0:3])

#Training and testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state=6)

#Create a model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#Predictions
y_pred = model.predict(x_test)

#Accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)