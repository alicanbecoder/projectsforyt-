# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:45:43 2025

@author: Alican
"""

import pandas as pd
import os

# 📂 Dosya yolları
doluluk_path = r"C:\Users\Alican\Desktop\Turizm-data\itosam-veri-istanbul-turizm-istatistikleri-otel-doluluk-orani.xlsx"
merged_path = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model\merged_aylik_macro_dataset.csv"
output_path = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model\final_aylik_dataset.csv"

# 🧾 Verileri oku
doluluk = pd.read_excel(doluluk_path)
merged = pd.read_csv(merged_path)

# 🚫 "Kaynak" veya geçersiz satırları temizle
doluluk = doluluk[doluluk["Yıl"].apply(lambda x: str(x).isdigit())]
doluluk["Yıl"] = doluluk["Yıl"].astype(int)

# 🔹 Ay isimlerini sayıya çevir
ay_map = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6,
    "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12
}
doluluk["Ay"] = doluluk["Dönem"].map(ay_map)
doluluk = doluluk.rename(columns={"Türkiye": "Otel_Doluluk_Turkiye"})
doluluk = doluluk[["Yıl", "Ay", "Otel_Doluluk_Turkiye"]]

# 🔧 Tip uyuşmazlığı gider
merged["Yıl"] = merged["Yıl"].astype(int)
merged["Ay"] = merged["Ay"].astype(int)

# 🔗 Doluluk oranını ekle
merged = merged.merge(doluluk, on=["Yıl", "Ay"], how="left")

# 🦠 Pandemi etkisi (2020-03 → 2021-06)
merged["Pandemi"] = merged.apply(
    lambda x: 1 if (x["Yıl"] == 2020 and x["Ay"] >= 3) or (x["Yıl"] == 2021 and x["Ay"] <= 6) else 0,
    axis=1
)

# 📆 Pandemi sonrası toparlanma (2021-07 sonrası)
merged["PostCovidTrend"] = merged.apply(
    lambda x: 1 if (x["Yıl"] > 2021) or (x["Yıl"] == 2021 and x["Ay"] >= 7) else 0,
    axis=1
)

# 🌦️ Mevsimsellik sütunu (Sezon)
def get_season(month):
    if month in [12, 1, 2]:
        return "Kış"
    elif month in [3, 4, 5]:
        return "İlkbahar"
    elif month in [6, 7, 8]:
        return "Yaz"
    else:
        return "Sonbahar"

merged["Sezon"] = merged["Ay"].apply(get_season)

# 🔢 Ay dummy değişkenleri (Ay_1–Ay_12)
ay_dummies = pd.get_dummies(merged["Ay"], prefix="Ay")
merged = pd.concat([merged, ay_dummies], axis=1)

# 🔢 Doluluk oranını yuvarla
if "Otel_Doluluk_Turkiye" in merged.columns:
    merged["Otel_Doluluk_Turkiye"] = merged["Otel_Doluluk_Turkiye"].round(2)

# 💾 Kaydet
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged.to_csv(output_path, index=False, encoding="utf-8-sig")

print("✅ Final veri seti başarıyla oluşturuldu!")
print(f"📁 Kayıt yeri: {output_path}")
print(merged.tail(12))


