# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:04:06 2025

@author: Alican
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Veri setini belirtilen yoldan yükleyelim
carseats = pd.read_csv(r'C:\Users\Alican\Desktop\Ödev\Carseats.csv')

# Modeli kurma (Sales ~ Price + Urban + US)
model = smf.ols('Sales ~ Price + Urban + US', data=carseats).fit()

# Modelin özetini yazdıralım
print("Model Özeti:")
print(model.summary())

# Katsayıları yorumlama
print("\nKatsayılar ve Yorumlar:")
print(model.params)

# Modelin denklem formu
print("\nModel Denklem Formu:")
print("Sales = β0 + β1 * Price + β2 * Urban[T.Y] + β3 * US[T.Y] + ε")

# Sıfır hipotezi testini yapma (p-değerleri)
print("\np-değerleri:")
print(model.pvalues)

# Anlamlı değişkenleri kullanarak sadeleştirilmiş model
model_sade = smf.ols('Sales ~ Price + US', data=carseats).fit()

# Yeni modelin özetini yazdıralım
print("\nSadeleştirilmiş Model Özeti:")
print(model_sade.summary())

# İki modelin uyumunu karşılaştıralım (R² ve Düzeltilmiş R²)
print("\nİki Modelin Uyum Karşılaştırması:")
print(f"İlk Model R²: {model.rsquared}")
print(f"İlk Model Düzeltilmiş R²: {model.rsquared_adj}")
print(f"Sadeleştirilmiş Model R²: {model_sade.rsquared}")
print(f"Sadeleştirilmiş Model Düzeltilmiş R²: {model_sade.rsquared_adj}")

# Sadeleştirilmiş modeldeki katsayılar için %95 güven aralıkları
print("\nSadeleştirilmiş Modelin Katsayıları için %95 Güven Aralıkları:")
print(model_sade.conf_int(alpha=0.05))

# Aykırı değerler ve yüksek etkiye sahip gözlemler için influence plot
print("\nAykırı Değerler ve Etkili Gözlemler için Influence Plot:")
sm.graphics.influence_plot(model_sade)
plt.show()

