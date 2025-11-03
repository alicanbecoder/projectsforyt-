# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 18:41:04 2025

@author: Alican
"""

import pandas as pd
import os

# 📂 Dosya yolları
yolcu_dosya = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model\turkiye_aylik_yolcu_sayisi.csv"
turizm_dosya = r"C:\Users\Alican\Desktop\Turizm-data\turizm geliri ve kisi basi ortalama harcama-2015-2024.xlsx"
cikis_klasoru = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model"
cikis_dosya = os.path.join(cikis_klasoru, "merged_aylik_dataset.csv")

# 🧾 Verileri oku
yolcu = pd.read_csv(yolcu_dosya)
turizm = pd.read_excel(turizm_dosya, header=1)  # ✅ Doğru satırdan başlıklar alındı

# 🧹 Sütun adlarını düzelt ve temizle
turizm.columns = turizm.columns.str.strip()
for col in turizm.columns:
    if turizm[col].dtype == object:
        turizm[col] = (
            turizm[col]
            .astype(str)
            .str.replace(" ", "")
            .str.replace(",", ".")
            .str.replace("-", "")
        )

# 🔢 Sayısal sütunları tanımla
num_cols = ["Ziyaretci_Turizm_Geliri_USD", "avg_Kisi_Basi_Harcama_USD", "Gecelik_Harcama_USD"]

# Sayısal dönüşüm ve yuvarlama
for col in num_cols:
    turizm[col] = pd.to_numeric(turizm[col], errors="coerce").round(0).astype("Int64")

# 🔄 Quarter → Ay eşlemesi
quarter_map = {"I": [1, 2, 3], "II": [4, 5, 6], "III": [7, 8, 9], "IV": [10, 11, 12]}

# 🔹 Çeyrek verisini 3 aya böl
turizm_aylik = []
for _, row in turizm.iterrows():
    yil = int(row["Yıl"])
    q = str(row["Quarter"]).strip()
    if q in quarter_map:
        for ay in quarter_map[q]:
            yeni = row.copy()
            yeni["Ay"] = ay
            for c in num_cols:
                yeni[c] = round(row[c] / 3) if pd.notnull(row[c]) else None
            turizm_aylik.append(yeni)

# Yeni DataFrame oluştur
turizm_aylik = pd.DataFrame(turizm_aylik)

# 🔗 Yıl ve Ay ile birleştir
merged = pd.merge(yolcu, turizm_aylik, on=["Yıl", "Ay"], how="left")

# Gereksiz sütunları at (örneğin Quarter ve Transfer gelir sütunu)
merged.drop(columns=["Quarter", "Transfer_Yolcu_Geliri_USD"], inplace=True, errors="ignore")

# 🔢 Sayısal sütunları tam sayıya yuvarla
numeric_cols = merged.select_dtypes(include=["float64", "int64"]).columns
merged[numeric_cols] = merged[numeric_cols].round(0).astype("Int64")

# 💾 CSV olarak kaydet
os.makedirs(cikis_klasoru, exist_ok=True)
merged.to_csv(cikis_dosya, index=False, encoding="utf-8-sig")

print("✅ Aylık birleşik veri seti başarıyla oluşturuldu!")
print(f"📁 Kayıt yeri: {cikis_dosya}")
print(merged.head(12))



