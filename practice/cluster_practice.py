# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:44:26 2017

@author: Nicole
"""

import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 3], 
             [1.2, 1.8],
             [7, 9],
             [1, 5],
             [1, 4.6], 
             [8, 10]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.show()