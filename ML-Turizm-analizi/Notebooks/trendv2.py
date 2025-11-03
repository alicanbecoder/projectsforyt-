# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:33:25 2025

@author: Alican
"""

from pytrends.request import TrendReq
import pandas as pd
import numpy as np  # np.nan kullanimi için eklendi
import time
import random

# 🌍 Odak ülkeler (Ayni)
countries = {
    "DE": "Germany",
    "US": "United States",
    "NL": "Netherlands",
    "GB": "United Kingdom",
    "IR": "Iran",
    "KZ": "Kazakhstan",
    "PL": "Poland",
    "RO": "Romania",
    "RU": "Russia",
    "SA": "Saudi Arabia"
}

# 🔑 Anahtar kelime seti (Ayni)
keywords = [
    # Genel Türkiye turizmi
    "Turkey travel", "Turkey holiday", "visit Turkey", "Turkey vacation",
    "Turkey tourism", "Turkey trip", "Turkey resorts", "Turkey beaches",
    "flights to Turkey", "Turkey all inclusive", "Turkey family holiday",

    # Şehir & bölge bazlı
    "Antalya holiday", "Antalya hotel", "Antalya resort", "Antalya beach",
    "Istanbul trip", "Istanbul travel", "Istanbul hotel", "Istanbul city break",
    "Cappadocia hot air balloon", "Cappadocia travel", "Cappadocia cave hotel",
    "Bodrum beach", "Bodrum hotel", "Fethiye holiday", "Marmaris hotel",
    "Izmir travel", "Alanya hotel", "Kusadasi resort",

    # Fiyat & sezon & rezervasyon
    "Turkey cheap hotels", "Turkey all inclusive resorts", "Turkey summer vacation",
    "best time to visit Turkey", "Turkey weather", "Turkey hotel deals",
    "Turkey flights", "Turkey visa", "Turkey e-visa",

    # Kültür & gastronomi
    "Turkish food", "Turkish culture", "Turkish coffee", "Istanbul shopping",
    "Grand Bazaar Istanbul", "Pamukkale", "Ephesus Turkey", "Turkish bath"
]

# 📍 YÖNTEM: Çapa (Anchor) Kelimeyi Belirle
# Bu kelime, tüm karsilastirmalar için temel alinacak.
anchor_keyword = "Turkey travel"

# Diğer kelimeleri çapadan ayır (4'lü gruplar halinde sorgulanacaklar)
other_keywords = [k for k in keywords if k != anchor_keyword]

# 🧠 TrendReq başlat
pytrends = TrendReq(hl='en-US', tz=360)
all_data = []

# 📆 Tarih aralığı
timeframe = '2022-01-01 2024-12-31'

# 🔄 Her ülke için veri çek
for code, name in countries.items():
    print(f"\n🌍 {name} ({code}) için veriler çekiliyor...\n")
    
    # 1. ADIM: Baz Puan (Baseline) Verisini Çek
    # Sadece çapa kelimenin trendini alarak ana ölçeğimizi oluşturuyoruz.
    print(f"  🔹 Baz puan alınıyor: [{anchor_keyword}]")
    df_baseline = None
    retries = 0
    success = False
    
    while not success and retries < 5:
        try:
            pytrends.build_payload([anchor_keyword], cat=67, timeframe=timeframe, geo=code)
            df_baseline = pytrends.interest_over_time()
            df_baseline = df_baseline.drop(columns=['isPartial'], errors='ignore').reset_index()
            
            # Sıfıra bölme hatasını engellemek için 0'ları np.nan ile değiştir
            df_baseline[f'{anchor_keyword}_safe'] = df_baseline[anchor_keyword].replace(0, np.nan)
            df_baseline['country'] = name
            
            # Bu, ülkenin ana veri çerçevesi olacak
            country_data = df_baseline.copy()
            success = True
            time.sleep(random.randint(5, 10))
            
        except Exception as e:
            retries += 1
            print(f"  ⚠️ BAZ PUAN hatası ({retries}. deneme): {e}")
            time.sleep(30)

    if not success:
        print(f"  ❌ {name} için baz puan alınamadı. Bu ülke atlanıyor.")
        continue

    # 2. ADIM: Diğer Kelimeleri 4'lü Gruplar Halinde Çek ve Ölçekle
    for i in range(0, len(other_keywords), 4):
        group_of_4 = other_keywords[i:i+4]
        query_group = [anchor_keyword] + group_of_4  # Çapa kelimeyi gruba ekle
        
        print(f"  🔹 {query_group} sorgulanıyor...")
        
        success = False
        retries = 0

        while not success and retries < 5:
            try:
                pytrends.build_payload(query_group, cat=67, timeframe=timeframe, geo=code)
                df_temp = pytrends.interest_over_time()
                df_temp = df_temp.drop(columns=['isPartial'], errors='ignore').reset_index()

                # Bu gruptaki çapa kelimenin 0'larını da nan yap
                df_temp[anchor_keyword] = df_temp[anchor_keyword].replace(0, np.nan)

                # 3. ADIM: NORMALİZASYON VE ÖLÇEKLEME
                # Bu bölüm, veriyi anlamlı hale getiren en kritik yerdir.
                
                # df_temp'i (geçici grup) df_baseline (baz puan) ile birleştir
                df_merged = pd.merge(
                    df_temp,
                    df_baseline[['date', anchor_keyword, f'{anchor_keyword}_safe']],
                    on='date',
                    suffixes=('_group', '_baseline')
                )
                
                # Ölçeklenmiş verileri tutmak için yeni bir dataframe
                df_scaled_batch = df_merged[['date']].copy()

                for kw in group_of_4:
                    if kw in df_merged.columns:
                        # 1. Göreceli Oran: Kelimenin, grup içindeki çapaya oranı
                        relative_ratio = df_merged[kw] / df_merged[f'{anchor_keyword}_group']
                        
                        # 2. Ölçeklenmiş Puan: Göreceli oranın, ana baz puan ile çarpımı
                        # (Burada _baseline olanı kullanıyoruz)
                        scaled_score = relative_ratio * df_merged[anchor_keyword + '_baseline']
                        
                        df_scaled_batch[kw] = scaled_score

                # Bu grubun ölçeklenmiş verilerini ana ülke verisiyle birleştir
                country_data = pd.merge(country_data, df_scaled_batch, on='date', how='outer')
                
                success = True
                time.sleep(random.randint(5, 10)) # API rate limit koruması
            except Exception as e:
                retries += 1
                print(f"  ⚠️ GRUP SORGUSU hatası ({retries}. deneme): {e}")
                time.sleep(30) # hata sonrası bekleme

    if not country_data.empty:
        all_data.append(country_data)

# 🧩 Tüm ülkeleri birleştir
print("\n🔄 Tüm veriler birleştiriliyor...")
final_df = pd.concat(all_data, ignore_index=True)

# Artık tüm kelimeler (çapa dahil) aynı ölçekte.
# Çapa kelimenin adını netleştirelim:
final_df = final_df.rename(columns={anchor_keyword: f"{anchor_keyword}_(Scaled)"})

# Güvenli (safe) sütunu temizleyelim
final_df = final_df.drop(columns=[f'{anchor_keyword}_safe'], errors='ignore')

final_df = final_df.sort_values(by=['country', 'date']).reset_index(drop=True)

# 4. ADIM: Anlamlı Ortalama Hesapla
# Artık tüm sütunlar aynı ölçekte olduğu için ortalama almak GEÇERLİDİR.
keyword_cols = [col for col in final_df.columns if col not in ['date', 'country']]
final_df['Average_Interest_SCALED'] = final_df[keyword_cols].mean(axis=1)

# 💾 Excel çıktısı
output_path = "google_trends_SCALED_2022_2024.xlsx" # Dosya adını değiştirdim
final_df.to_excel(output_path, index=False)
print("\n✅ Tüm veriler başarıyla çekildi, ölçeklendi ve kaydedildi!")
print(f"📁 Kaydedilen dosya: {output_path}") 