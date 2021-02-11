#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
# Decision Tree Regressor model
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor  
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
import pydotplus
from subprocess import check_call
import graphviz

dataset = np.array( 
[['Asset Flip', 100, 1000], 
['Text Based', 500, 3000], 
['Visual Novel', 1500, 5000], 
['2D Pixel Art', 3500, 8000], 
['2D Vector Art', 5000, 6500], 
['Strategy', 6000, 7000], 
['First Person Shooter', 8000, 15000], 
['Simulator', 9500, 20000], 
['Racing', 12000, 21000], 
['RPG', 14000, 25000], 
['Sandbox', 15500, 27000], 
['Open-World', 16500, 30000], 
['MMOFPS', 25000, 52000], 
['MMORPG', 30000, 80000] 
]) 

# select all rows by : and column 1 
# by 1:2 representing features 
X= dataset[:,1:2].astype(int) #covert to integer
print(X)

# select all rows by : and column 2 
# by 2 to Y representing labels 
y = dataset[:, 2].astype(int)  
print(y)

reg = DecisionTreeRegressor(random_state=0)
print(reg)
reg.fit(X,y)

pred_case = reg.predict([[3750]])
# print the predicted price 
print("Predicted price: % d\n"% pred_case)  

#Visualize results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
# plot predicted data 
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')  
  
# specify title 
plt.title('Profit to Production Cost (Decision Tree Regression)')  
# specify X axis label 
plt.xlabel('Production Cost') 
  
# specify Y axis label 
plt.ylabel('Profit') 
  
# show the plot 
plt.show() 

#generate decision tree visualization
dotfile = export_graphviz (reg, out_file = None, feature_names = ['Production Cost'])

graph = graphviz.Source(dotfile)
graph.render('dtree_render', view=True)


