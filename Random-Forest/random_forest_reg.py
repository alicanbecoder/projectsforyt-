# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:11:51 2024

@author: Alican
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Örnek veri oluşturma
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Karar ağacı regresyon modelini oluşturma
tree_model = DecisionTreeRegressor(max_depth=3)
tree_model.fit(X, y)

# Karar ağacını görselleştirme
plt.figure(figsize=(12,8))
plot_tree(tree_model, filled=True)
plt.title("Karar Ağacının Görselleştirilmesi")
plt.show()

# Veriyi oluşturma
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest regresyon modelini oluşturma
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Linear Regresyon modelini oluşturma
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_rf = rf_model.predict(X_test)  # Random Forest tahminleri
y_pred_lr = linear_model.predict(X_test)  # Linear Regresyon tahminleri

# Bireysel ağaçların tahminleri (Random Forest için)
y_pred_trees = np.array([tree.predict(X_test) for tree in rf_model.estimators_]).T

# Görselleştirme
plt.figure(figsize=(12, 8))

# Gerçek değerler
plt.scatter(X_test, y_test, color='black', label='Gerçek Değerler')

# Bireysel ağaçların tahminleri
for i in range(10):  # İlk 10 ağacın tahminlerini çizelim
    plt.plot(X_test, y_pred_trees[:, i], linestyle='--', alpha=0.5, label=f'Ağaç {i+1}' if i < 1 else "")

# Nihai tahmin (Random Forest'in ortalaması)
plt.plot(X_test, y_pred_rf, color='red', label='Random Forest Tahmini', linewidth=2)

# Linear Regresyon tahminini çizme
plt.plot(X_test, y_pred_lr, color='blue', label='Linear Regresyon Tahmini', linestyle='-', linewidth=2)

# Grafik başlığı ve etiketler
plt.title("Random Forest ve Linear Regresyon Tahminleri")
plt.xlabel("Özellik (X)")
plt.ylabel("Hedef (y)")
plt.legend()

# Görselleştirmeyi gösterme
plt.show()

# Nihai tahmin (Random Forest'in ortalaması)
plt.plot(X, y, color='red', label='x-y', linewidth=2)


# Hata hesaplama
mse = mean_squared_error(y_test, y_pred_rf)

# Hata dağılımı
plt.figure(figsize=(8, 6))
plt.hist(y_test - y_pred_rf, bins=30, color='blue', alpha=0.7, label='Tahmin Hataları')
plt.title(f"Hata Dağılımı (MSE: {mse:.2f})")
plt.legend()
plt.show()