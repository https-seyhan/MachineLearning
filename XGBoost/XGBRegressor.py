#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
import pandas as pd
import numpy as np
import xgboost
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
# 
X_enc = X_enc.apply(LabelEncoder().fit_transform) # Encode target labels with value between 0 and n_classes-1.

#print("X_enc :", X_enc)

mergedata = mergedata.drop(X_cat.columns, axis=1)
# #END LABEL ENCODING BLOCK
print("Mergedata Columns : ",mergedata.columns)
print("X_cat Columns ;", X_cat.columns)
# 
# =============================================================================

FinalData = pd.concat([mergedata,X_enc], axis=1)
print("Final Data Columns : ", FinalData.columns)
train = FinalData[:count]
test = FinalData[count:]

trainy = train['charges'].astype('int')
trainx = train.drop(['charges'], axis=1)
test = test.drop(['charges'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size= 0.3)

# Paramater list in a dictionary
param_dist = {'n_estimators':200, 'learning_rate':0.01, 'max_depth':7}

clf = xgboost.XGBRegressor(**param_dist)
clf.fit(X_train, y_train, eval_metric='logloss')

#print("Model :", clf)

#evals_result = clf.evals_result()

#print(evals_result)

y_testpred = clf.predict(X_test)
y_pred = clf.predict(test)

dftestpred = pd.DataFrame(y_testpred)

dfpred = pd.DataFrame(y_pred)
#print(dfpred)

rms = sqrt(mean_squared_error(y_test, y_testpred))

print("RMSE:", rms)


