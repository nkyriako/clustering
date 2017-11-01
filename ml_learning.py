# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 21:08:12 2017

@author: Nicole

More machine learning stuff. Use K-means clustering to categorize and
then assign the categories later 

"""

import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
        
        
    
'''
Parsing
'''

Text = open("movement_libras.data", "r")
Array = []
Classes = []
counter = 0;
for line in Text:
    abcissa_x = []
    abcissa_y = []
    Class=0
    if (len(line) != 0):
        for num in line.split(","):
            counter = counter + 1
            if(counter%2==0):
                abcissa_y.append(float(num))
            elif (counter == 91):
                Class=int(num)
                Classes.append(Class)
                counter = 0
            else: 
                abcissa_x.append(float(num))
    Array.append([abcissa_x, abcissa_y]) #, Class might be useful later 

#XL is for making 45 different clustering things and compare all of them               
#XL = []
#for i in range(45):
#    listx = []
#    listy = []
#    for [x, y] in Array:
#        listx.append(x[i])
#        listy.append(y[i])
#    XL.append([listx, listy])
#
#    A = np.array(XL[0])
#    xl = XL[i]
#    plt.scatter(xl[0][:], xl[1][:], s=15)
#    plt.title("Clusters: "+str(i+1))
#    plt.show()
#    clf = KMeans(n_clusters=15, random_state=1)
#    
    


#X = np.array([x[0] for x in Array])

#make like a sum of features to condense into one dimension 
nuArray = []
for pair in Array:
    nuPair = []
    for x in pair:
        rt = 0
        for num in x:
            rt = rt + num
        nuPair.append(rt)
    nuArray.append(nuPair)
    
X = np.array(nuArray)
#for group in Array:
#    plt.scatter(group[0], group[1], s=15)
#    plt.show()

'''
Plotting 
'''

plt.scatter(X[:,0], X[:,1], s=15)
plt.show()

clf = KMeans(n_clusters=15, random_state=1)
clf.fit(X)


''' 
Predicting K means
'''

# compare clusters to class results 

labels = clf.labels_
print("Labels: ")
for i in range(len(Classes)):
    print("Actual Class: ", Classes[i], "   Prediction: ", labels[i])

# this function was copied/tweaked from http://marcharper.codes/2016-07-11/Clustering+with+Scikit-Learn.html
def set_colors(labels, colors="r g b c m y k w C1 C2 C3 C4 C5 C6 C7"):
    colored_labels = []
    for label in labels: 
        colored_labels.append(colors.split(" ")[label])
    return(colored_labels)

colors = set_colors(labels)
plt.scatter(X[:,0], X[:,1], c=colors, s=15)
plt.xlabel("x")
plt.ylabel("y")
plt.title("K MEANS CLUSTERS")
plt.show()
#notes: for some reason the labels change ever time and then so do the clusters


'''
Spectral Clustering 
'''

sc = SpectralClustering(n_clusters=15, random_state=1)
sc.fit(X)
labels = sc.labels_

colors = set_colors(labels)
plt.scatter(X[:,0], X[:,1], c=colors, s=15)
plt.xlabel("x")
plt.ylabel("y")
plt.title("SPECTRAL CLUSTERS")
plt.show()


'''
Training separately 
'''



