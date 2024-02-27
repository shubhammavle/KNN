# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:27:13 2024

@author: HP
"""

import pandas as pd
import numpy as np

glass = pd.read_csv('E:/datasets/glass.csv')
glass.head()

glass.columns
glass.describe
glass.info()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(np.array(glass.iloc[:,:-1]), np.array(glass.iloc[:,-1]), test_size=0.2)

X_train.shape
y_train.shape

np.array(glass.iloc[:,:-1]).shape
np.array(glass.iloc[:,-1]).shape

acc = []
max_acc = -1
k = -1
for i in range(3,50,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    train_acc = np.mean(knn.predict(X_train)==y_train)
    test_acc = np.mean(knn.predict(X_test)==y_test)
    if test_acc>max_acc:
        max_acc=test_acc
        k = i    
print(max_acc)

knn = KNeighborsClassifier(n_neighbors=k)
np.mean(knn.predict(X_test)==y_test)