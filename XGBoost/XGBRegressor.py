#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
#XGBoost Regression with Decision Trees
import pandas as pd
import numpy as np
import xgboost # XGBoost module
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Encode target labels with value between 0 and n_classes-1
from math import sqrt
from sklearn.metrics import mean_squared_error

os.chdir('/home/saul/pythonWork/files')
data = pd.read_csv('insurance.csv')
testdata = pd.read_csv('insurance_test.csv')

mergedata = data.append(testdata) # Append testdata to insurance data
testcount = len(testdata)

count = len(mergedata)-testcount
X_cat = mergedata.copy()
X_cat = mergedata.select_dtypes(include=['object'])
#print("X_cat :", X_cat)
X_enc = X_cat.copy()

#=============================================================================
# #LABEL ENCODING BLOCK
X_enc = X_enc.apply(LabelEncoder().fit_transform) # Encode target labels with value between 0 and n_classes-1.
#print("X_enc :", X_enc)
mergedata = mergedata.drop(X_cat.columns, axis=1)
##END LABEL ENCODING BLOCK
print("Mergedata Columns : ",mergedata.columns)
print("X_cat Columns ;", X_cat.columns)

# =============================================================================
FinalData = pd.concat([mergedata,X_enc], axis=1)
print("Final Data Columns : ", FinalData.columns)
train = FinalData[:count]
test = FinalData[count:]

trainy = train['charges'].astype('int')
trainx = train.drop(['charges'], axis=1)
test = test.drop(['charges'], axis=1)
#Cut training and Test Data
X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size= 0.3)

# Paramater list in a dictionary
param_dist = {'n_estimators':200, 'learning_rate':0.01, 'max_depth':7}
insurance_xg_regression = xgboost.XGBRegressor(**param_dist)
insurance_xg_regression.fit(X_train, y_train, eval_metric='logloss')

#print("Model :", insurance_xg_regression)
#evals_result = insurance_xg_regression.evals_result()
y_test_pred = insurance_xg_regression.predict(X_test)
y_pred = insurance_xg_regression.predict(test)

df_test_pred = pd.DataFrame(y_test_pred)
df_pred = pd.DataFrame(y_pred)

rms_xgboost = sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE:", rms_xgboost)
