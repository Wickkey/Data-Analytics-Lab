# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:42:27 2021

@author: VIGNESH S
"""
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


X,y = make_classification(n_features = 2, n_informative = 2, n_redundant=0)
class1 = y==0 
class2 = y==1 

clf = DecisionTreeClassifier()
clf.fit(X,y)


#plt.style.use(['science','ieee','grid','no-latex'])


plot_decision_regions(X, y, clf)
plt.grid(False)
plt.show()






