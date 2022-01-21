# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:36:54 2021

@author: VIGNESH S
"""
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

x1 = np.random.multivariate_normal([0,0],[[1,0.01],[0.01,0.98]], size = 50)
y1 = np.zeros((50,1))
x2 = np.random.multivariate_normal([1,1],[[0.5,0.01],[0.01,0.75]], size = 50)
X = np.concatenate((x1,x2))
y2 = np.ones((50,1))
y = np.concatenate((y1,y2))
y = y.astype(np.int_)
y = y.flatten()


plt.style.use(['science','ieee','grid','no-latex'])

from sklearn.naive_bayes import GaussianNB 
clf = GaussianNB()
clf.fit(X,y)
plot_decision_regions(X, y, clf)
plt.grid(False)
plt.show()

