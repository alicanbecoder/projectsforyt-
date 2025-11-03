# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:59:15 2025

@author: Alican
"""

import pandas as pd
import matplotlib.pyplot as plt

#  Dosya yolu
path = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model\final_aylik_dataset_clean.csv"

#  Veriyi oku
df = pd.read_csv(path)

#  Sadece sayısal değişkenleri al
num_df = df.select_dtypes(include=["float64", "int64"])

# Korelasyon matrisi
corr = num_df.corr().round(2)

# Korelasyon tablosunu yazdır
print("📈 Korelasyon Matrisi (sayısal değişkenler arası):")
print(corr["Dis_Hat_Yolcu_Sayisi"].sort_values(ascending=False))

plt.figure(figsize=(10,8))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Korelasyon Katsayısı")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Değişkenler Arası Korelasyon Isı Haritası")
plt.tight_layout()
plt.show()
