# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 20:06:49 2025

@author: Alican
"""

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


path = r"C:\Users\Alican\Desktop\Turizm-data\Machine-learning-model\final_aylik_dataset.csv"
df = pd.read_csv(path)

# Gereksiz sütunları at
df = df.drop(columns=["Otel_Doluluk_Turkiye"], errors="ignore")

#  Özellik ve hedef değişkenler
features = [
    "USD_TRY", "EUR_TRY", "TUFE_yillik_pct",
    "Pandemi", "PostCovidTrend", "Yıl", "Ay"
]

targets = {
    "Dis_Hat_Yolcu_Sayisi": "Dis_Hat_Yolcu_Sayisi",
    "Turizm_Geliri_USD": "Turizm_Geliri_USD"
}

#  Train / Test ayrımı (2024 test)
train = df[df["Yıl"] < 2024]
test = df[df["Yıl"] == 2024]

X_train = train[features]
X_test = test[features]

# Modeller (hepsi NaN destekli)
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, random_state=42, max_depth=10
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_depth=5, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, random_state=42, max_depth=5, learning_rate=0.1
    )
}

#  Sonuçları depolama
results = []

for label, target in targets.items():
    print(f"\n====================== {label} ======================")
    y_train = train[target]
    y_test = test[target]
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Target": label,
            "R2": round(r2, 3),
            "MAE": round(mae, 2)
        })
        print(f"{name:<20} | R²: {r2:.3f} | MAE: {mae:,.0f}")

# Sonuç tablosu
res_df = pd.DataFrame(results)
print("\n📈 Model Karşılaştırma Sonuçları")
print(res_df)


