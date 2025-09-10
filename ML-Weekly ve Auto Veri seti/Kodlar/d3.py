# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:27:20 2025

@author: Alican
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/selva86/datasets/master/Weekly.csv"
weekly = pd.read_csv(url)

# İlk satırda sütun isimleri düzgün ayrılmamışsa düzelt
weekly['Year'] = pd.to_numeric(weekly['Year'], errors='coerce')

# Giriş ve hedef değişkenleri ayır (örnek: Lag2 üzerinden)
X = weekly[['Lag2']]
y = weekly['Direction'].map({'Up': 1, 'Down': 0})

# Train (80%), Validation (10%), Test (10%) bölme
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.2, random_state=102, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=102, stratify=y_temp)

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# KNN için farklı k değerlerini test et
k_values = range(1, 101)
valid_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_full)
    valid_pred = knn.predict(X_valid_scaled)
    test_pred = knn.predict(X_test_scaled)
    valid_accuracies.append(accuracy_score(y_valid, valid_pred))
    test_accuracies.append(accuracy_score(y_test, test_pred))

best_k_index = test_accuracies.index(max(test_accuracies))
best_k = k_values[best_k_index]

plt.figure(figsize=(10, 6))
plt.plot(k_values, valid_accuracies, label='Validation Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='x')
plt.axvline(best_k, color='gray', linestyle='--', label=f'Best k = {best_k}')
plt.xlabel('k Değeri')
plt.ylabel('Doğruluk')
plt.title('KNN: k Değerine Göre Validation ve Test Doğruluğu')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"En iyi k (test doğruluğuna göre): {best_k}")
print(f"Validation doğruluğu (best_k için): {valid_accuracies[best_k_index]:.3f}")
print(f"Test doğruluğu: {test_accuracies[best_k_index]:.3f}")

# Modeli tanımla ve eğit (scaled train set)
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train_full)

# Tahminler
valid_pred_logreg = logreg.predict(X_valid_scaled)
test_pred_logreg = logreg.predict(X_test_scaled)

# Doğruluklar
valid_acc_logreg = accuracy_score(y_valid, valid_pred_logreg)
test_acc_logreg = accuracy_score(y_test, test_pred_logreg)

print(f"Lojistik Regresyon Validation Doğruluğu: {valid_acc_logreg:.3f}")
print(f"Lojistik Regresyon Test Doğruluğu: {test_acc_logreg:.3f}")