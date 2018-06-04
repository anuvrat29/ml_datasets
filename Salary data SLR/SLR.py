# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 11:42:48 2018

@author: testuser
"""

#Step1: Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step2: Load the dataset
dataset = pd.read_csv('Salary_Data.csv')

#Step3: Split the dataset with features and labels
features = dataset.iloc[:,0].values
label = dataset.iloc[:,1].values

features = features.reshape(-1,1)
label = label.reshape(-1,1)

#Step4: Handle the Missing data (if applicable) 
#Not required since there exists no missing data
#Step5: Deal with different types of data (Categorical Data)
#Not Required since there exists no categorical data
#Step6: Create training and testing set
#Training set to have 80% of the total data
#Testing set will have 20%

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
                                                 features,
                                                 label,
                                                 test_size=0.2,
                                                 random_state=10
                                                 )

#7. Feature Scaling

#Step8: Create a model

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# ValueError: Found arrays with inconsistent 
# numbers of samples: [ 1 24]
# To solve this bug, you need to use numpy's reshape method
X_train = X_train.reshape(24,1)
y_train = y_train.reshape(24,1)
X_test = X_test.reshape(6,1)
y_test = y_test.reshape(6,1)
#model.fit(feature_train_data,label_train_data)
model.fit(X_train,y_train)

#Step9: Test your model
y_prediction = model.predict(X_test)

#Step10: Plot the graph
plt.scatter(X_test,y_test,color = 'r')
plt.plot(X_test,y_prediction, color = 'b')

plt.scatter(X_train,y_train,color = 'r')
plt.plot(X_train,model.predict(X_train), color = 'b')


from sklearn.metrics import r2_score
r2_score(y_test,y_prediction)
#0.98
#r2_score(actual_label,predicted_label)
r2_score(y_train,model.predict(X_train))
#0.94
#Training data R2 score: 0.95 ---- Testing R2: 0.87
# My tolerating range is +/- 6%
 
#Two perspective:
#1. Since my result % is more than train result %
#   either the model is over-fitted or its normal

#2. If my training % > test % , for sure the model
# is underfitted.

#Example of K-cross validation where k =5

from sklearn.cross_validation import cross_val_score
#Invoke the regressor object if not invoked before
#cross_val_score(model,feature,label,kvalue)
cv_results = cross_val_score(model,features,label,cv=2)
print(cv_results)
print(np.mean(cv_results))






from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_prediction)

print('c value')
model.intercept_
print('m value')
model.coef_



#My Final Equation is 
# label = 9356.86299354(feature) + 26089.09663242