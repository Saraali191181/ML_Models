# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:22:23 2022

@author: hp
"""

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

# import the data

customer_data =  pd.read_csv('shopping-data.csv')

customer_data.shape

#execute the head() function of the data frame
customer_data.head()

# the first three columns from our dataset

data = customer_data.iloc[:, 3:5].values

#create the dendrograms for our dataset

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))


# group the data points into the  five clusters
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

#plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
