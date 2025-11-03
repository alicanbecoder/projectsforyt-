# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 19:03:15 2025

@author: Alican
"""

import pandas as pd
import os

# 📂 Klasör yolu
base_path = r"C:\Users\Alican\Desktop\Turizm-data\Euro-dolar-datası"

# 🔹 Dosya yolları
evds_path = os.path.join(base_path, "EVDS (1).xlsx")
enflasyon_path = os.path.join(base_path, "Enflasyon.xlsx")

# 🧾 EVDS verisini oku
evds = pd.read_excel(evds_path)
print("📊 EVDS (Döviz) verisi sütunları:")
print(evds.columns.tolist())
print(evds.head(10), "\n")

# 🧾 Enflasyon verisini oku
enf = pd.read_excel(enflasyon_path)
print("📈 Enflasyon verisi sütunları:")
print(enf.columns.tolist())
print(enf.head(10))
