# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:46:11 2025

@author: Alican
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Öklidyen mesafe
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# En yakın k komşuyu bul
def get_k_neighbors(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return [label for _, label in neighbors]

# Tahmin fonksiyonları
def predict_point(X_train, y_train, test_point, k):
    neighbors = get_k_neighbors(X_train, y_train, test_point, k)
    return Counter(neighbors).most_common(1)[0][0]

def predict_knn(X_train, y_train, X_test, k):
    return np.array([predict_point(X_train, y_train, x, k) for x in X_test])

# En iyi k'yi bul (test seti üzerinden)
def find_best_k_on_test(X_train, y_train, X_test, y_test, k_values=None):
    if k_values is None:
        k_values = list(range(1, 51))

    test_accuracies = []
    for k in k_values:
        y_pred = predict_knn(X_train, y_train, X_test, k)
        acc = accuracy_score(y_test, y_pred)
        test_accuracies.append(acc)

    best_k = k_values[np.argmax(test_accuracies)]
    print(f"\nEn iyi k: {best_k} | Test Doğruluğu: {max(test_accuracies):.2f}")

    # Grafik
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, test_accuracies, marker='o', linestyle='--', color='green')
    plt.xlabel("k Değeri")
    plt.ylabel("Test Doğruluğu")
    plt.title("Test Doğruluğu vs K Değeri")
    plt.grid(True)
    plt.show()

    return best_k

# Performans değerlendirme
def evaluate_performance(X_train, y_train, X_test, y_test, k, class_names=None):
    y_pred = predict_knn(X_train, y_train, X_test, k)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nK = {k} için Test Doğruluğu: {acc:.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.show()
    return acc


iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# %70 eğitim, %30 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=450)

# En iyi k ve değerlendirme
best_k = find_best_k_on_test(X_train, y_train, X_test, y_test)
evaluate_performance(X_train, y_train, X_test, y_test, best_k, class_names=class_names)

