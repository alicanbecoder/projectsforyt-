# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:59:42 2025

@author: Alican
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Veri setini indir ve yükle
url = "https://raw.githubusercontent.com/selva86/datasets/master/Weekly.csv"
weekly = pd.read_csv(url)

# Year sütununu sayıya çevir (genelde int ama garanti için)
weekly['Year'] = pd.to_numeric(weekly['Year'], errors='coerce')

# Eğitim ve test seti oluştur (1990-2008 eğitim, 2009-2010 test)
train = weekly[weekly['Year'] < 2009]
test = weekly[weekly['Year'] >= 2009]

# Özellik ve hedef değişken - sadece Lag2 kullanılıyor
X_train = train[['Lag2']]
y_train = train['Direction']
X_test = test[['Lag2']]
y_test = test['Direction']

# --- a) Lojistik Regresyon ---
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

cm_logreg = confusion_matrix(y_test, y_pred_logreg)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print("Lojistik Regresyon Karışıklık Matrisi:\n", cm_logreg)
print("Lojistik Regresyon Doğruluk Oranı:", acc_logreg)

# --- d) K-NN (K=1) ---
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("\nK-NN (K=1) Karışıklık Matrisi:\n", cm_knn)
print("K-NN Doğruluk Oranı:", acc_knn)

# --- f) Naive Bayes ---
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

cm_nb = confusion_matrix(y_test, y_pred_nb)
acc_nb = accuracy_score(y_test, y_pred_nb)
print("\nNaive Bayes Karışıklık Matrisi:\n", cm_nb)
print("Naive Bayes Doğruluk Oranı:", acc_nb)




