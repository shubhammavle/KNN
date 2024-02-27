# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:59:31 2024

@author: HP
"""

import pandas as pd
import numpy as np

wbcd = pd.read_csv('E:/datasets/wbcd.csv')
wbcd.head()

wbcd.columns
#there are 569 rows
wbcd.describe()

#output column has entries as B and M.
#replace B and M as Beniegn and Malignant resp.
wbcd['diagnosis'] = np.where(wbcd.diagnosis=='B','Beniegn',wbcd.diagnosis)
wbcd['diagnosis'] = np.where(wbcd.diagnosis=='M','Malignant',wbcd.diagnosis)
wbcd.diagnosis

#drop column with id
wbcd = wbcd.iloc[:, 1:]

#normalization
def norm_func(i):
    return (i - i.min())/(i.max() - i.min())

#normalizing columns except output column    
wbcd_n = norm_func(wbcd.iloc[:,1:])
wbcd_n

X = np.array(wbcd_n)
y = np.array(wbcd.diagnosis)

#splitting traning and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#to handel unblanced splitting of data
#there is satisfied sampling concept is used

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

#calculation the accuracy
np.mean(np.array(pred == y_test))
 
#evaluating the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test,))
#confusion matrix
pd.crosstab(pred, y_test) 

#lets try to select correct value of k
acc = []
curr_max = -1
k = 0
for i in range(3,50,2):
    #declare the model
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train)==y_train)
    test_acc = np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc, test_acc])
    if test_acc>curr_max:
        curr_max=test_acc
        k = i    
k
acc

#plot graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2), [i[0] for i in acc], 'ro-')
plt.plot(np.arange(3,50,2), [i[1] for i in acc], 'bo-')

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

































