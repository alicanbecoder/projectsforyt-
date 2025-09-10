# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:38:14 2024

@author: Alican
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def create_dataset(samples=400, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=100):
    """Veri kümesini oluştur ve eğitim/val/test setlerine ayır."""
    
    # Veri setini oluşturma
    X, y = make_moons(n_samples=samples, noise=0.2, random_state=90)
    
    # Rasgele karıştırma ve eğitim setini ayırma
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_ratio, random_state=random_state)
    
    # Kalan seti doğrulama ve test olarak ayırma
    val_split = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_split, random_state=random_state)
    
    return X, y, X_train, X_val, X_test, y_train, y_val, y_test


def plot_dataset(X, y):
    """Veri kümesini görselleştir."""
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.title("Veri Kümesi")
    plt.xlabel("Özellik 1")
    plt.ylabel("Özellik 2")
    plt.legend()
    plt.show()
    return


    
    


    
    