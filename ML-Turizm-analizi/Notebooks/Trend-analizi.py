
from pytrends.request import TrendReq
import pandas as pd
import time
import random

# 🌍 Odak ülkeler ve yerel arama terimleri
country_keywords = {
    "DE": {  # Germany
        "country": "Germany",
        "keywords": ["Türkei Urlaub", "Türkei Reise"]
    },
    "GB": {  # United Kingdom
        "country": "United Kingdom",
        "keywords": ["Turkey holiday", "Turkey travel"]
    },
    "US": {  # United States
        "country": "United States",
        "keywords": ["Turkey vacation", "Turkey travel"]
    },
    "RU": {  # Russia
        "country": "Russia",
        "keywords": ["отдых в Турции", "поездка в Турцию"]
    },
    "NL": {  # Netherlands
        "country": "Netherlands",
        "keywords": ["vakantie Turkije", "reizen naar Turkije"]
    },
    "IR": {  # Iran
        "country": "Iran",
        "keywords": ["سفر به ترکیه", "تور ترکیه"]
    },
    "PL": {  # Poland
        "country": "Poland",
        "keywords": ["wakacje w Turcji", "podróż do Turcji"]
    },
    "RO": {  # Romania
        "country": "Romania",
        "keywords": ["vacanță în Turcia", "călătorie în Turcia"]
    },
    "KZ": {  # Kazakhstan
        "country": "Kazakhstan",
        "keywords": ["Түркия демалысы", "Түркияға сапар"]
    },
    "SA": {  # Saudi Arabia
        "country": "Saudi Arabia",
        "keywords": ["السفر إلى تركيا", "عطلة تركيا"]
    }
}

# 📆 Tarih aralığı
timeframe = '2022-01-01 2024-12-31'

# 🧠 TrendReq başlat
pytrends = TrendReq(hl='en-US', tz=360)
all_data = []

# 🔄 Her ülke için veri çek
for code, info in country_keywords.items():
    country = info["country"]
    kw_list = info["keywords"]
    print(f"\n🌍 {country} ({code}) için veriler çekiliyor: {kw_list}\n")

    success = False
    retries = 0

    while not success and retries < 5:
        try:
            pytrends.build_payload(kw_list, cat=67, timeframe=timeframe, geo=code)
            df = pytrends.interest_over_time().reset_index()
            df = df.drop(columns=['isPartial'], errors='ignore')
            df['country'] = country
            all_data.append(df)
            success = True
            time.sleep(random.randint(5, 10))  # Rate limit koruma
        except Exception as e:
            retries += 1
            print(f"⚠️ {retries}. denemede hata: {e}")
            time.sleep(30)

# 📊 Tüm verileri birleştir
final_df = pd.concat(all_data, ignore_index=True)
# Ortalama ilgi değeri ekle
keyword_cols = [col for col in final_df.columns if col not in ['date', 'country']]
final_df['Average_Interest'] = final_df[keyword_cols].mean(axis=1)
final_df = final_df.sort_values(by=['country', 'date']).reset_index(drop=True)

# 💾 Excel çıktısı
output_path = "google_trends_localized_turkey_interest_2022_2024.xlsx"
final_df.to_excel(output_path, index=False)

print("\n✅ Yerelleştirilmiş veriler başarıyla çekildi ve kaydedildi!")
print(f"📁 Dosya adı: {output_path}")
print(final_df.head())

