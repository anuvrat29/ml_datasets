# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("cars.csv")
features = dataset.iloc[:,1:].values
label = dataset.iloc[:,0].values

# Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode = LabelEncoder()
features[:,1] = encode.fit_transform(features[:,1])

hotencode = OneHotEncoder(categorical_features=[1])
features = hotencode.fit_transform(features).toarray()

# Training and testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.25,random_state=30)

# Create a model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

# Predictions
y_pred = model.predict(x_test)

# Accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)