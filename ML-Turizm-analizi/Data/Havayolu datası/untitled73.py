# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:50:03 2025

@author: Alican
"""

import pandas as pd
import re
import glob
import os

# --- 1. Klasör ve dosya kalıbı (Senin Lokal Yolun) ---
excel_klasor_yolu = r"C:\Users\Alican\Desktop\Çalışma grubu\Havayolu datası\2023\pdf24_convertPdfTo (1)"
dosya_kalibi = os.path.join(excel_klasor_yolu, "YOLCU_2023_*.xlsx")

# --- 2. Ay isimleri ---
aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
         "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

# --- 3. Boş DataFrame oluştur ---
ana_df = pd.DataFrame()

# --- 4. Dosyaları bul ve sırala ---
bulunan_dosyalar = sorted(glob.glob(dosya_kalibi))

if not bulunan_dosyalar:
    print(f"UYARI: '{excel_klasor_yolu}' klasöründe .xlsx dosyası bulunamadı.")
else:
    print(f"Toplam {len(bulunan_dosyalar)} adet .xlsx dosyası bulundu.")

# --- 5. Dosyaları sırayla oku ve BİRLEŞTİR (MERGE) ---
for dosya_yolu in bulunan_dosyalar:
    
    ay_numarasi_match = re.search(r'_(\d{1,2})\.xlsx$', dosya_yolu)
    
    if not ay_numarasi_match:
        print(f"Uyarı: '{os.path.basename(dosya_yolu)}' adında ay numarası bulunamadı, atlanıyor.")
        continue
        
    ay_numarasi = int(ay_numarasi_match.group(1))
    ay_adi = aylar[ay_numarasi - 1]
    print(f"İşleniyor: {os.path.basename(dosya_yolu)} (Ay: {ay_adi})")

    try:
        # --- Veri Okuma ---
        df = pd.read_excel(dosya_yolu, skiprows=3, header=None)
        
        # --- DIŞ HAT SÜTUNU (İndeks 5 / 6. Sütun) ---
        df = df.iloc[:, [0, 5]]
        df.columns = ["Havalimanı", ay_adi]

        # --- TEMİZLİK 1: Boş Havalimanı Satırlarını At (Dip Toplamlar) ---
        df = df.dropna(subset=['Havalimanı'])

        # --- TEMİZLİK 2 (İsim Standardizasyonu) ---
        df['Havalimanı'] = (
            df['Havalimanı'].astype(str)
            .str.replace(r"[\*(),]", "", regex=True)  # *, (), , karakterlerini sil
            .str.replace(r"\s+", "", regex=True)       # Tüm boşlukları sil
            .str.upper()                              # Tümünü büyük harf yap
            .str.strip()                              # Ekstra baş/son boşlukları temizle
        )
        
        # 'TOPLAM' gibi temizlikten kalan gereksiz satırları at
        df = df[~df['Havalimanı'].str.contains("TOPLAM", na=False)]
        # Not satırını da burada atayabiliriz
        df = df[~df['Havalimanı'].str.contains("YILIÇERISINDE", na=False)]

        # --- TEMİZLİK 3: Veri Sütunu (Kars 'NaN' sorunu için) ---
        df[ay_adi] = df[ay_adi].astype(str)
        df[ay_adi] = df[ay_adi].str.replace("nan", "", case=False)
        df[ay_adi] = df[ay_adi].str.replace("None", "", case=False)
        df[ay_adi] = (
            df[ay_adi]
            .str.replace(r"[^\d]", "", regex=True) # Nokta vb. kaldır
            .replace("", "0") # Boş kalanları '0' yap
            .astype(float)
        )
        
        # --- BİRLEŞTİRME (MERGE) ---
        if ana_df.empty:
            ana_df = df
        else:
            ana_df = pd.merge(
                ana_df,
                df,
                on="Havalimanı",  # Anahtar (Artık temiz ve standart)
                how="outer"       # Farklı havalimanı listeleri için (66 vs 67)
            )
        
    except Exception as e:
        print(f"HATA: '{dosya_yolu}' işlenirken sorun oluştu: {e}")

# --- 6. Birleştirme Sonrası Temizlik ---
islenen_aylar = [col for col in ana_df.columns if col in aylar]
ana_df[islenen_aylar] = ana_df[islenen_aylar].fillna(0)


# --- 7. Aylık veriyi kümülatiften ayır ---
if not ana_df.empty:
    print(f"\nAylık veri ayrıştırması yapılıyor (Aylar: {islenen_aylar})...")

    aylik_df = ana_df.copy()
    
    for i in range(len(islenen_aylar) - 1, 0, -1): 
        ay_simdiki = islenen_aylar[i]
        ay_onceki = islenen_aylar[i-1]
        aylik_df[ay_simdiki] = aylik_df[ay_simdiki] - aylik_df[ay_onceki]

    # --- 8. Negatifleri temizle ---
    aylik_df.loc[:, islenen_aylar] = aylik_df.loc[:, islenen_aylar].clip(lower=0)

    # --- 9. Toplam sütunu ekle ---
    aylik_df["Toplam (Aylık Dış Hat)"] = aylik_df[islenen_aylar].sum(axis=1)
    
    
    # --- 10. İSTENMEYEN SATIRLARI DROPLA (YENİ EKLENDİ) ---
    print("\nİstenmeyen 'TOPLAM' ve 'NOT' satırları sonuçtan çıkarılıyor...")
    # Temizlenmiş (büyük harf, boşluksuz) halleriyle liste:
    drop_list = [
        "TÜRKİYEGENELİ",
        "TÜRKİYEGENELİDİREKTTRANSİT",
        "TÜRKİYEGENELİDİREKTTRANSİTDAHİL",
        "YILIÇERISINDEGEÇMIŞAYLARDAYAPILANREVIZELERMEVCUTAYVERILERINEYANSITILMIŞTIR.",
        "DİĞERDİREKTTRANSİT",
        "DHMİDİREKTTRANSİT"
    ]
    
    # 'Havalimanı' sütunu bu listede OLMAYAN satırları tut
    aylik_df_filtrelenmis = aylik_df[~aylik_df['Havalimanı'].isin(drop_list)]
    
    
    # --- 11. Havalimanı adına göre sırala ---
    aylik_df_sonuc = aylik_df_filtrelenmis.sort_values(by="Havalimanı").reset_index(drop=True)

    
    # --- 12. Excel olarak kaydet (Aynı klasöre) ---
    cikti_dosya_yolu = os.path.join(excel_klasor_yolu, "SONUC_AYLIK_DIS_HAT.xlsx")
    try:
        aylik_df_sonuc.to_excel(cikti_dosya_yolu, index=False)
        print(f"\n--- BAŞARILI ---")
        print(f"Filtrelenmiş sonuç dosyası kaydedildi: {cikti_dosya_yolu}")
        
        print("\n--- KONTROL (Kars Satırı) ---")
        kars_satiri = aylik_df_sonuc[aylik_df_sonuc['Havalimanı'].str.contains('KARS', na=False)]
        if kars_satiri.empty:
            print("Kars satırı bulunamadı.")
        else:
            print(kars_satiri.to_string())
        
    except Exception as e:
        print(f"\nHATA: Dosya '{cikti_dosya_yolu}' konumuna kaydedilemedi: {e}")
else:
    print("\nVeri işlenemedi (Muhtemelen dosya okuma hatası).")