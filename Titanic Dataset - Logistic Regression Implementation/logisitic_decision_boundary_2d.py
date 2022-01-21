# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:28:56 2021

@author: VIGNESH S
"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

x,y = make_classification(n_features = 2, n_informative = 2, n_redundant=0,random_state=68)
class1 = y==0 
class2 = y==1 


clf = LogisticRegression(penalty = 'none')
clf.fit(x,y)
x1_coef,x2_coef = clf.coef_[0][0],clf.coef_[0][1]
bias =  clf.intercept_
a = np.linspace(-3,2)
ahat = -x1_coef*a/(x2_coef) + bias/(-x2_coef)


plt.style.use(['ieee','grid','no-latex'])
plt.scatter(x[class1][:,0],x[class1][:,1],label = 'class1')
plt.scatter(x[class2][:,0],x[class2][:,1],label='class2')
plt.plot(a,ahat,'b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Example of Logistic Regression')
plt.legend()
plt.savefig('logisticexample')
plt.close()

u = np.linspace(-10,10)
sigmoid = 1/(1+np.exp(-u))
proba = (x @ clf.coef_.T + bias)
plt.plot(u,sigmoid,'b')
plt.scatter(proba[class1],[0]*sum(class1), s = 2, label = 'class1')
plt.scatter(proba[class2],[1]*sum(class2), s =2, label = 'class2')
plt.axvline(x=0,color = 'g', linestyle = '--')
plt.legend() 
plt.xlabel('w$^T$x + b$_{0}$')
plt.ylabel('Y')
plt.title('Sigmoid Function')
plt.savefig('sigmoidfunction')








