# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:53:17 2020

@author: Rituparna
"""

############## California Housing Price Prediction ###################

import pandas as pd
import os

os.getcwd()

##################### 1. Load the Data ###############################

Datas = pd.read_excel("Desktop\\Simplilearn\\Python\\Projects\\California Housing\\California_Housing.xlsx")
Datas.to_csv('Desktop\\Simplilearn\\Python\\Projects\\California Housing\\California_Housing.csv', index = None, header = True, sep = ',')
Datas

Datas.columns

x = Datas.iloc[:, 0:9]                    # Assigning regressor variables
x

y = Datas['median_house_value']           # Assigning regressand varable
y

##################### 2. Handling Missing Values #######################

x.isnull().any()

# Out[61]: 
# longitude             False
# latitude              False
# housing_median_age    False
# total_rooms           False
# total_bedrooms         True
# population            False
# households            False
# median_income         False
# ocean_proximity       False
# dtype: bool

# So, in this database variable 'total_bedrooms' has missing values

x.isnull().any().sum()

# Out[75]: 1
# There is a single missing value in the column

# Imputing the missing values with the mean of the total_bedrooms

x.total_bedrooms = x.total_bedrooms.fillna(x.total_bedrooms.mean()) 

x.isnull().any()

# Out[79]: 
# longitude             False
# latitude              False
# housing_median_age    False
# total_rooms           False
# total_bedrooms        False
# population            False
# households            False
# median_income         False
# ocean_proximity       False
# dtype: bool

y.isnull().any()

# Out[80]: False
# Variable 'median_house_value' doesn't have any missing value

##################### 3. Encoding Categorical Data #######################

Datas.dtypes

# Variable 'ocean_proximity' is a categorical variable

print(x.ocean_proximity.unique())

# print(x.ocean_proximity.unique())
# ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']

# Mapping the categorical variable into a numeric variable

x.ocean_proximity = x.ocean_proximity.map({'NEAR BAY' : 0, '<1H OCEAN' : 1,
                                          'INLAND' : 2, 'NEAR OCEAN' : 3, 'ISLAND' : 4})

print(x.ocean_proximity.unique())

# print(x.ocean_proximity.unique())
# [0 1 2 3 4]


####################### 4. Splitting the Dataset #########################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)

print(x_train.shape)
print(x_test.shape)

# print(x_train.shape)
# print(x_test.shape)
# (16512, 9)
# (4128, 9)

print(y_train.shape)
print(y_test.shape)

# print(y_train.shape)
# print(y_test.shape)
# (16512,)
# (4128,)


######################## 5. Standardizing Data ###########################

from sklearn import preprocessing

print(x_train)
x_train_std = preprocessing.scale(x_train)
print(x_train_std[0:5])

# print(x_train_std[0:5])
# [[ 1.07002458 -0.78929505  0.0311079  -1.15459867 -1.23002776 -1.20746458
#   -1.25882715  4.56876242  0.62543849]
# [ 1.26459455 -1.39310608  0.26880051  0.73590997  0.591564    1.24750538
#    0.77478346  0.05203806  1.79553144]
# [ 1.50406529 -0.84546352 -0.84043166  2.45685801  2.66144611  0.51505911
#    1.10455815 -0.25083298  0.62543849]
# [ 0.63099489 -0.76589152  1.21957095  0.64606128  0.89715213  0.17477495
#    0.91349821 -0.7230252  -0.54465446]
# [-1.81360223  2.29060928 -0.12735383 -0.48695377 -0.42069669 -0.50667265

print(x_test)
x_test_std = preprocessing.scale(x_test)
print(x_test_std)

print(y_train)
y_train_std = preprocessing.scale(y_train)
print(y_train_std)

print(y_test)
y_test_std = preprocessing.scale(y_test)
print(y_test_std)


##################### 6. Perform Linear Regression ########################

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Building the model

California_model = LinearRegression().fit(x_train_std, y_train_std) 

R_Square = California_model.score(x_train_std, y_train_std)
print(R_Square)

# print(R_Square)
# 0.6363389416297167
# Explains 63.6% of the variation in the response variable around its mean.
# So, it is a good fit.


Y0 = California_model.intercept_
print(Y0)

#  print(Y0)
# -2.7989661062160142e-15

Beta_coef = California_model.coef_
print(Beta_coef)

# print(Beta_coef)
# [-0.72980033 -0.77941303  0.12080491 -0.09983307  0.26026453 -0.39816867
#   0.27682221  0.64595218 -0.02400719]

for i, j in zip(list(x_train.columns), California_model.coef_):
    print(i, ' , ', j)

# longitude  ,  -0.7298003345766108
# latitude  ,  -0.7794130315614274
# housing_median_age  ,  0.12080490537167503
# total_rooms  ,  -0.09983307430484019
# total_bedrooms  ,  0.2602645318100054
# population  ,  -0.39816866787803756
# households  ,  0.27682220769816385
# median_income  ,  0.6459521812034931
# ocean_proximity  ,  -0.024007188834955746
    
# Inference:
# Coefficient values indicates as follows:
# for a unit change in longitude, there is a decreases 
# of 0.7396 unit of Median_house_value
# the maximum increase in median_house_value of 0.6498 unit 
# happens when there is a unit change in median_income.
    
y_predicted =  California_model.predict(x_test_std)  # Testing the model
print(y_predicted)

df = pd.DataFrame({'Actual': y_test_std, 'Predicted': y_predicted})
print(df)

# print(df)
#         Actual  Predicted
# 0     0.227553  -0.354633
# 1     1.061897   0.762718
# 2    -0.468168   0.149468
# 3    -0.669173  -0.702590
# 4     2.536521   2.386608
#        ...        ...
# 4123  0.113188   0.025980
# 4124 -1.133564  -0.143249
# 4125 -0.988008  -1.302636
# 4126 -0.807797  -0.412957
# 4127 -0.217778   0.320450

# [4128 rows x 2 columns]

# Calculation of Root Mean Squared Error for the model

RMSE = np.sqrt(metrics.mean_squared_error(y_test_std, y_predicted))
print(RMSE)

# print(RMSE)
# 0.604616028315269


########## 7. Linear Regression with One Independent Variable ###########

x1 = pd.DataFrame(x['median_income'])
print(x1)

# Split dataset

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = .2)

print(x1_train.shape, x1_test.shape)
print(y_train.shape, y_test.shape)

# print(x1_train.shape, x1_test.shape)
# print(y_train.shape, y_test.shape)
# (16512, 1) (4128, 1)
# (16512,) (4128,)

# Scale the data

xtrain_std = preprocessing.scale(x1_train)
xtest_std = preprocessing.scale(x1_test)
ytrain_std = preprocessing.scale(y_train)
ytest_std = preprocessing.scale(y_test)

# Train the model

model_2 = LinearRegression().fit(xtrain_std, ytrain_std)

r_sqr = model_2.score(xtrain_std, ytrain_std)
print(r_sqr)

# print(r_sqr)
# 0.4662852257396791

y0 = model_2.intercept_
print(y0)

#  print(y0)
# -1.4194067640223588e-16

beta_coe = model_2.coef_
print(beta_coe)

#  print(beta_coe)
# [0.68285081]

# Test the model

y_pred = model_2.predict(xtest_std)

pf = pd.DataFrame({"Actual Y": ytest_std, "Predicted Y": y_pred})
print(pf)

#  print(pf)
#       Actual Y  Predicted Y
# 0     0.190033     0.074518
# 1    -0.292522    -0.249439
# 2    -0.798984     0.054319
# 3    -0.944193    -0.354200
# 4    -1.209819    -0.579120
# ...        ...          ...
# 4123  0.622119    -1.093823
# 4124 -1.389560    -0.716171
# 4125  1.507541    -0.627269
# 4126 -0.727264    -0.512978
# 4127 -0.738775     0.486166

# [4128 rows x 2 columns]

Rmse = np.sqrt(metrics.mean_squared_error(ytest_std, y_pred))
print(Rmse)

# print(Rmse)
# 0.7046236024739734

################# Plot Train and Test Data ############################

import matplotlib.pyplot as plt

Test = ytest_std[0:50]
Train = y_pred[0:50]

def Compare_plot():
    plt.plot(Test)
    plt.plot(Train)
    plt.xlabel("No. of Data")
    plt.ylabel("Regressand Value")
    plt.title("Comparison of Train and Test Data")
    plt.legend(["Test Data", "Train Data"], loc = 2)
    return

Compare_plot()

##########################################################################







