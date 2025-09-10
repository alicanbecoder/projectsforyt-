# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:32:29 2025

@author: Alican
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükleme ve ön hazırlık
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
auto = pd.read_csv(url, na_values='?').dropna()

median_mpg = auto['mpg'].median()
auto['high_mpg'] = (auto['mpg'] > median_mpg).astype(int)

X = auto.select_dtypes(include=[np.number]).drop(columns=['mpg'])
y = auto['high_mpg']

# Parametre değerleri
C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1, 10]
degree_values = [2, 3, 4, 5]

# Linear Kernel doğrulukları
linear_acc = []
for c in C_values:
    svm = SVC(C=c, kernel='linear')
    score = cross_val_score(svm, X, y, cv=5).mean()
    linear_acc.append({'C': c, 'Accuracy': score})
linear_df = pd.DataFrame(linear_acc)

# RBF Kernel doğrulukları
rbf_results = []
for c in C_values:
    for g in gamma_values:
        svm = SVC(C=c, kernel='rbf', gamma=g)
        score = cross_val_score(svm, X, y, cv=5).mean()
        rbf_results.append({'C': c, 'Gamma': g, 'Accuracy': score})
rbf_df = pd.DataFrame(rbf_results)

# Polynomial Kernel doğrulukları
poly_results = []
for c in C_values:
    for d in degree_values:
        svm = SVC(C=c, kernel='poly', degree=d)
        score = cross_val_score(svm, X, y, cv=5).mean()
        poly_results.append({'C': c, 'Degree': d, 'Accuracy': score})
poly_df = pd.DataFrame(poly_results)

# Linear Kernel Grafik
plt.figure(figsize=(8,6))
plt.plot(linear_df['C'], linear_df['Accuracy'], marker='o')
plt.title("Linear Kernel: C vs Accuracy")
plt.xscale('log')
plt.xlabel("C Değeri")
plt.ylabel("Doğruluk")
plt.grid(True)
plt.show()

# RBF Kernel Heatmap
plt.figure(figsize=(8,6))
rbf_pivot = rbf_df.pivot(index='Gamma', columns='C', values='Accuracy')
sns.heatmap(rbf_pivot, annot=True, fmt=".4f", cmap='coolwarm')
plt.title("RBF Kernel: C vs Gamma - Accuracy")
plt.xlabel("C Değeri")
plt.ylabel("Gamma Değeri")
plt.show()

# Polynomial Kernel Heatmap
plt.figure(figsize=(8,6))
poly_pivot = poly_df.pivot(index='Degree', columns='C', values='Accuracy')
sns.heatmap(poly_pivot, annot=True, fmt=".4f", cmap='viridis')
plt.title("Polynomial Kernel: C vs Degree - Accuracy")
plt.xlabel("C Değeri")
plt.ylabel("Degree")
plt.show()


