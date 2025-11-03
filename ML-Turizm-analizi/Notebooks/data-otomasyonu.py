# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 20:15:01 2025

@author: Alican
"""

import pandas as pd
import re
import glob
import os

# --- 1. Global Ayarlar ve Sabitler ---

# İşlenecek ay isimleri
aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
         "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

# Sonuçtan atılacak istenmeyen satırların (temizlenmiş) listesi
drop_list = [
    "TÜRKİYEGENELİ",
    "TÜRKİYEGENELİDİREKTTRANSİT",
    "TÜRKİYEGENELİDİREKTTRANSİTDAHİL",
    "YILIÇERISINDEGEÇMIŞAYLARDAYAPILANREVIZELERMEVCUTAYVERILERINEYANSITILMIŞTIR.",
    "DİĞERDİREKTTRANSİT",
    "DHMİDİREKTTRANSİT",
    "DHMİTOPLAMI",  # <-- YENİ EKLENDİ
    "TOPLAM" # 'TOPLAM' içerenleri de atar
]


# --- 2. Tek bir Yılı İşleyecek Ana Fonksiyon ---
def process_year(year_to_process, base_folder_path):
    """
    Belirtilen tek bir yılı (örn: 2015) alır,
    tüm aylık dosyaları okur, temizler, birleştirir,
    hesaplar ve sonucu kendi klasörüne kaydeder.
    """
    print(f"--- YIL {year_to_process} BAŞLADI ---")
    
    # 1. O yıla ait klasör yolunu ve dosya kalıbını oluştur
    excel_klasor_yolu = os.path.join(base_folder_path, str(year_to_process), "pdf24_convertPdfTo (1)")
    dosya_kalibi = os.path.join(excel_klasor_yolu, f"YOLCU_{year_to_process}_*.xlsx")
    
    # 2. Dosyaları bul
    bulunan_dosyalar = sorted(glob.glob(dosya_kalibi))
    
    if not bulunan_dosyalar:
        print(f"UYARI: Yıl {year_to_process} için '{excel_klasor_yolu}' klasöründe .xlsx dosyası bulunamadı. Atlanıyor.")
        return # Bu yılı atla, fonksiyondan çık

    print(f"{year_to_process} için {len(bulunan_dosyalar)} adet dosya bulundu. İşleniyor...")
    
    ana_df = pd.DataFrame() # Her yıl için sıfırdan bir ana_df başlat

    # 3. Dosyaları sırayla oku ve BİRLEŞTİR (MERGE)
    for dosya_yolu in bulunan_dosyalar:
        
        ay_numarasi_match = re.search(r'_(\d{1,2})\.xlsx$', dosya_yolu)
        if not ay_numarasi_match:
            continue
            
        ay_numarasi = int(ay_numarasi_match.group(1))
        ay_adi = aylar[ay_numarasi - 1]

        try:
            df = pd.read_excel(dosya_yolu, skiprows=3, header=None)
            df = df.iloc[:, [0, 5]]
            df.columns = ["Havalimanı", ay_adi]
            df = df.dropna(subset=['Havalimanı'])

            # TEMİZLİK 1: Havalimanı İsimleri (En Önemli Kısım)
            df['Havalimanı'] = (
                df['Havalimanı'].astype(str)
                .str.replace(r"[\*(),]", "", regex=True)
                .str.replace(r"\s+", "", regex=True)
                .str.upper()
                .str.strip()
            )
            
            # TEMİZLİK 2: Veri Sütunu (Kars 'NaN' sorunu)
            df[ay_adi] = df[ay_adi].astype(str)
            df[ay_adi] = df[ay_adi].str.replace("nan", "", case=False)
            df[ay_adi] = df[ay_adi].str.replace("None", "", case=False)
            df[ay_adi] = (
                df[ay_adi]
                .str.replace(r"[^\d]", "", regex=True)
                .replace("", "0")
                .astype(float)
            )
            
            # BİRLEŞTİRME (MERGE)
            if ana_df.empty:
                ana_df = df
            else:
                ana_df = pd.merge(ana_df, df, on="Havalimanı", how="outer")
            
        except Exception as e:
            print(f"HATA: '{dosya_yolu}' işlenirken sorun oluştu: {e}")

    # --- Yıl için tüm aylar birleştirildi, şimdi hesaplama ---
    if ana_df.empty:
        print(f"HATA: Yıl {year_to_process} için veri okunamadı. Atlanıyor.")
        return

    # 4. Birleştirme Sonrası Temizlik (NaN -> 0)
    islenen_aylar_bu_yil = [col for col in ana_df.columns if col in aylar]
    ana_df[islenen_aylar_bu_yil] = ana_df[islenen_aylar_bu_yil].fillna(0)

    # 5. Kümülatiften Aylığa Çevirme
    print(f"Yıl {year_to_process} için aylık veriler hesaplanıyor...")
    aylik_df = ana_df.copy()
    for i in range(len(islenen_aylar_bu_yil) - 1, 0, -1): 
        ay_simdiki = islenen_aylar_bu_yil[i]
        ay_onceki = islenen_aylar_bu_yil[i-1]
        aylik_df[ay_simdiki] = aylik_df[ay_simdiki] - aylik_df[ay_onceki]

    # 6. Negatifleri temizle
    aylik_df.loc[:, islenen_aylar_bu_yil] = aylik_df.loc[:, islenen_aylar_bu_yil].clip(lower=0)

    # 7. Toplam sütunu ekle
    aylik_df[f"Toplam {year_to_process} (Aylık Dış Hat)"] = aylik_df[islenen_aylar_bu_yil].sum(axis=1)
    
    # 8. İstenmeyen Satırları Dropla
    print("İstenmeyen 'TOPLAM' ve 'NOT' satırları çıkarılıyor...")
    # 'Havalimanı' sütunu 'drop_list' içinde OLMAYAN satırları tut
    aylik_df_filtrelenmis = aylik_df[~aylik_df['Havalimanı'].isin(drop_list)]
    
    # 9. Sırala
    aylik_df_sonuc = aylik_df_filtrelenmis.sort_values(by="Havalimanı").reset_index(drop=True)

    # 10. Excel olarak kaydet
    cikti_dosya_yolu = os.path.join(excel_klasor_yolu, f"SONUC_AYLIK_DIS_HAT_{year_to_process}.xlsx")
    try:
        aylik_df_sonuc.to_excel(cikti_dosya_yolu, index=False)
        print(f"--- BAŞARILI: Yıl {year_to_process} ---")
        print(f"Sonuç dosyası kaydedildi: {cikti_dosya_yolu}\n")
        
    except Exception as e:
        print(f"\nHATA: Dosya '{cikti_dosya_yolu}' konumuna kaydedilemedi: {e}\n")


# --- 3. Ana Otomasyon Yöneticisi ---
def main():
    
    # Ana klasör yolu (Yıl klasörlerini içeren)
    base_folder = r"C:\Users\Alican\Desktop\Turizm-data\Havayolu datası"
    
    # Hangi yılları işlemek istediğiniz (2025 dahil DEĞİL)
    years_to_process = list(range(2015, 2025)) # 2015, 2016, ..., 2024
    
    print("--- OTOMASYON BAŞLADI ---")
    print(f"İşlenecek yıllar: {years_to_process}")
    
    for year in years_to_process:
        try:
            process_year(year, base_folder)
        except Exception as e:
            print(f"!!! Yıl {year} işlenirken BEKLENMEDİK BİR HATA oluştu: {e} !!!")
            # Bir yılda hata olursa durma, sonrakine geç
            pass
            
    print("--- OTOMASYON TAMAMLANDI ---")

# Bu script'i doğrudan çalıştırdığınızda 'main' fonksiyonunu çağır
if __name__ == "__main__":
    main()