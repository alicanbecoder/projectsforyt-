# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 16:41:35 2025

@author: Alican
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt


base_path = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model"
data_path = os.path.join(base_path, "final_aylik_dataset_clean.csv")
model_path = os.path.join(base_path, "rf_yolcu_model.pkl")


rf = joblib.load(model_path)
print("✅ Model yüklendi!")


df = pd.read_csv(data_path)
df = df.drop(columns=["Otel_Doluluk_Turkiye"], errors="ignore")

features = ["USD_TRY","EUR_TRY","TUFE_yillik_pct","Pandemi","PostCovidTrend","Yıl","Ay"]

# =====================================================
#  2024 TAHMİNİ (Model + %15 Artışlı Yaz Ayları)
# =====================================================
test_2024 = df[df["Yıl"] == 2024].copy()
pred_2024 = rf.predict(test_2024[features])

test_2024["Tahmin_RF"] = pred_2024

# Belirli aylarda %15 artış uygula
special_months = [4,5,6,7,8,9,11,12]
test_2024["Tahmin_RF"] = test_2024.apply(
    lambda row: row["Tahmin_RF"] * 1.15 if row["Ay"] in special_months else row["Tahmin_RF"],
    axis=1
)

# Toplam yolcu tahmini
total_2024_pred = test_2024["Tahmin_RF"].sum()
print(f"🛫 2024 Tahmini Toplam Yolcu Sayısı (ay düzeltmeli): {total_2024_pred:,.0f}")

# =====================================================
#  2025 Forecast — 3 Senaryo (İyimser / Nötr / Kötümser)
# =====================================================
months = list(range(1, 13))

scenarios = {
    "İyimser": {
        "USD_TRY": np.linspace(31, 33, 12),
        "EUR_TRY": np.linspace(33, 35, 12),
        "TUFE_yillik_pct": np.linspace(35, 25, 12),
    },
    "Nötr": {
        "USD_TRY": np.linspace(33, 36, 12),
        "EUR_TRY": np.linspace(35, 38, 12),
        "TUFE_yillik_pct": np.linspace(35, 30, 12),
    },
    "Kötümser": {
        "USD_TRY": np.linspace(36, 39, 12),
        "EUR_TRY": np.linspace(38, 42, 12),
        "TUFE_yillik_pct": np.linspace(40, 35, 12),
    }
}

results = {}

for name, vals in scenarios.items():
    forecast_2025 = pd.DataFrame({
        "Yıl": [2025]*12,
        "Ay": months,
        "USD_TRY": vals["USD_TRY"],
        "EUR_TRY": vals["EUR_TRY"],
        "TUFE_yillik_pct": vals["TUFE_yillik_pct"],
        "Pandemi": [0]*12,
        "PostCovidTrend": [1]*12
    })
    
    # 🔹 Model tahmini
    forecast_2025["Tahmin_RF"] = rf.predict(forecast_2025[features])
    
    # 🔹 Belirli aylarda %15 artış uygula
    forecast_2025["Tahmin_RF"] = forecast_2025.apply(
        lambda row: row["Tahmin_RF"] * 1.15 if row["Ay"] in special_months else row["Tahmin_RF"],
        axis=1
    )
    
    total_2025 = forecast_2025["Tahmin_RF"].sum()
    results[name] = total_2025
    print(f"🔹 {name} Senaryosu 2025 Toplam Yolcu Sayısı (ay düzeltmeli): {total_2025:,.0f}")

# =====================================================
# 2024 vs 2025 Senaryo Karşılaştırma Grafiği
# =====================================================
labels = ["2024 Tahmini", "2025 İyimser", "2025 Nötr", "2025 Kötümser"]
values = [
    total_2024_pred,
    results["İyimser"],
    results["Nötr"],
    results["Kötümser"]
]

# Yüzdelik farkları hesapla
pct_changes = [(v - total_2024_pred) / total_2024_pred * 100 for v in values[1:]]

plt.figure(figsize=(9,5))
bars = plt.bar(labels, values, color=["#4caf50","#81c784","#ffb74d","#e57373"])
plt.title("✈️ Dış Hat Yolcu Sayısı — 2024 ve 2025 Senaryoları (%15 Ay Düzeltmeli)")
plt.ylabel("Toplam Yolcu Sayısı")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Bar etiketleri
for i, bar in enumerate(bars):
    height = bar.get_height()
    if i == 0:
        plt.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                 f"{height:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                 f"{height:,.0f}\n({pct_changes[i-1]:+.2f}%)", ha='center', va='bottom', fontsize=9)

plt.show()


