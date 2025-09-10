# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:26:31 2025

@author: Alican
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi indir
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
auto = pd.read_csv(url, na_values='?').dropna()

# Medyan mpg hesapla
median_mpg = auto['mpg'].median()
print(f"Medyan MPG: {median_mpg:.2f}")

# Binary sınıf oluştur
auto['high_mpg'] = (auto['mpg'] > median_mpg).astype(int)

# Genel görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(data=auto, x='mpg', hue='high_mpg', bins=30, kde=True, palette='Set1')
plt.axvline(median_mpg, color='black', linestyle='--', label=f'Median mpg = {median_mpg:.2f}')
plt.title('MPG Dağılımı ve Sınıf Etiketi (high_mpg)')
plt.xlabel('Miles Per Gallon (mpg)')
plt.ylabel('Frekans')
plt.legend()
plt.tight_layout()
plt.show()

# 0 ve 1 ler
plt.figure(figsize=(6, 4))
sns.countplot(x='high_mpg', data=auto, palette='Set2')
plt.xticks([0, 1], ['Low MPG (0)', 'High MPG (1)'])
plt.title('Sınıf Dağılımı: High MPG Etiketi')
plt.xlabel('Sınıf')
plt.ylabel('Adet')
plt.tight_layout()
plt.show()