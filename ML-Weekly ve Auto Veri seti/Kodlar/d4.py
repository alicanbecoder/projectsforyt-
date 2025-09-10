# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:52:28 2025

@author: Alican
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
auto = pd.read_csv(url, na_values='?').dropna()

median_mpg = auto['mpg'].median()
auto['high_mpg'] = (auto['mpg'] > median_mpg).astype(int)

X = auto.select_dtypes(include=[np.number]).drop(columns=['mpg'])
y = auto['high_mpg']

C_values = [0.01, 0.1, 1, 10,100]
gamma_values = [0.001, 0.01, 0.1, 1, 10]
degree_values = [2, 3, 4, 5]

print("Linear Kernel:")
for c in C_values:
    svm = SVC(C=c, kernel='linear')
    scores = cross_val_score(svm, X, y, cv=5)
    print(f"C = {c}, Ortalama Doğruluk: {scores.mean():.4f}")
    
print("RBF Kernel için C-Gamma kombinasyonlarının doğrulukları:\n")
for c in C_values:
    for g in gamma_values:
        svm = SVC(C=c, kernel='rbf', gamma=g)
        scores = cross_val_score(svm, X, y, cv=5)
        print(f"C = {c:<5} Gamma = {g:<5} => Ortalama Doğruluk: {scores.mean():.4f}")
        
print("Polynomial Kernel için C-Degree kombinasyonlarının doğrulukları:\n")
for c in C_values:
    for d in degree_values:
        svm = SVC(C=c, kernel='poly', degree=d)
        scores = cross_val_score(svm, X, y, cv=5)
        print(f"C = {c:<5} Degree = {d:<2} => Ortalama Doğruluk: {scores.mean():.4f}")





