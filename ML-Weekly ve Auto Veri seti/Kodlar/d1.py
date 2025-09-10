# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:54:19 2025

@author: Alican
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Kolon isimleri
column_names = ['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Direction']

# Dosyayı oku
weekly = pd.read_csv(
    r"C:\Users\Alican\Desktop\Ödev\cok final\4a3904f9-1fdb-4b38-9cdd-85453ee4110e\Weekly.csv",
    skiprows=1,
    names=column_names
)

# Year sütununu sayısal değere çevir
weekly['Year'] = pd.to_numeric(weekly['Year'], errors='coerce')

# Eksik değerleri at (isteğe bağlı)
weekly = weekly.dropna(subset=['Year', 'Lag2', 'Direction'])

# Eğitim ve test verisi oluştur
train = weekly[weekly['Year'] < 2009]
test = weekly[weekly['Year'] >= 2009]

X_train = train[['Lag2']]
y_train = train['Direction']
X_test = test[['Lag2']]
y_test = test['Direction']

# KNN (k=1)
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

# Naive Bayes
nb = GaussianNB().fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))

