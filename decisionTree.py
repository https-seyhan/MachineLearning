#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""

# Compare Linear Regression with Decision Tree Regressor
import numpy as np
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor # decision tree regressor
from matplotlib import pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1,1)   

reg = DecisionTreeRegressor(min_samples_split=3).fit(X,y)
plt.figure()
plt.plot(line, reg.predict(line), label ="decision tree")

reg = LinearRegression().fit(X,y)

plt.plot(line, reg.predict(line), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
