# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:46:26 2022

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read data
bankdata = pd.read_csv("bill_authentication.csv")
#To see the rows and columns and of the data
bankdata.shape
#To get a feel of how our dataset actually looks
bankdata.head()

#To divide the data into attributes and labels
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

#divide data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Training  the Algorithm
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

# Making Predictions

y_pred = svclassifier.predict(X_test)

#Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))








