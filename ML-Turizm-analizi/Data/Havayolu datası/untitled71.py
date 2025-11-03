# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 03:23:29 2025

@author: Alican
"""

import pandas as pd
import numpy as np
import re

# --- Dosya Yolunu Belirt ---
# Örnek olarak 2015 Ocak dosyasının yolu
dosya_yolu = r"C:\Users\Alican\Desktop\PATH_TO_YOUR_FILE\YOLCU_2015_01.xlsx" # Kendi yolunu yazmalısın

print(f"İşleniyor: {dosya_yolu}")

try:
    # --- Adım 1: Dosyayı MultiIndex Başlıkla Oku ---
    # İlk satırı atla (skiprows=1)
    # Sonraki İKİ satırı başlık olarak kullan (header=[0, 1])
    df_raw = pd.read_excel(dosya_yolu, skiprows=1, header=[0, 1])
    
    # --- Adım 2: Sütun Başlıklarını Anlaşılır Hale Getir ---
    # MultiIndex başlıkları birleştir (Örn: ('2015 YILI OCAK AYI', 'Dış Hat') -> '2015_Dış Hat')
    yeni_sutunlar = []
    mevcut_yil_blok = ""
    for col1, col2 in df_raw.columns:
        # İlk seviye başlığı temizle (Yıl bilgisini al)
        col1_temiz = str(col1).strip()
        yil_mac = re.search(r'(\d{4})', col1_temiz)
        if yil_mac:
            mevcut_yil_blok = yil_mac.group(1) # Yılı güncelle
            
        # İkinci seviye başlığı temizle (Hat bilgisi)
        col2_temiz = str(col2).strip().replace(" ", "_")
        
        # Sütun adını oluştur
        if 'Unnamed:' not in col1_temiz and 'Unnamed:' not in col2_temiz : # İlk sütun (Havalimanları)
             yeni_sutunlar.append(col1_temiz) # Sadece 'Havalimanları'
        elif mevcut_yil_blok and col2_temiz:
             yeni_sutunlar.append(f"{mevcut_yil_blok}_{col2_temiz}")
        else:
             yeni_sutunlar.append(f"Gecici_{col1_temiz}_{col2_temiz}") # Eşleşmezse geçici isim
             
    df_raw.columns = yeni_sutunlar
    print("-> Sütun başlıkları düzeltildi:", yeni_sutunlar)

    # --- Adım 3: İstenen Sütunları Seç (Havalimanları ve Hedef Yıl Dış Hat) ---
    # Hedef yılı (ikinci yıl bloğu) başlık listesinden bul
    hedef_yil = ""
    if len(yeni_sutunlar) > 4 and yeni_sutunlar[4].startswith('20'): # Genellikle 5. sütun 2. yılla başlar
        hedef_yil = yeni_sutunlar[4].split('_')[0] 
        
    if not hedef_yil:
        raise ValueError("Hedef yıl (ikinci yıl bloğu) sütun başlıklarından tespit edilemedi.")

    havalimani_sutunu = 'Havalimanları'
    hedef_dis_hat_sutunu = f"{hedef_yil}_Dış_Hat"

    if havalimani_sutunu not in df_raw.columns or hedef_dis_hat_sutunu not in df_raw.columns:
         raise ValueError(f"'{havalimani_sutunu}' veya '{hedef_dis_hat_sutunu}' sütunları bulunamadı.")

    df_secilen = df_raw[[havalimani_sutunu, hedef_dis_hat_sutunu]].copy()
    print(f"-> Sütunlar seçildi: '{havalimani_sutunu}', '{hedef_dis_hat_sutunu}'")

    # --- Adım 4: Gereksiz Satırları ve Veriyi Temizle ---
    # İlk satır genellikle boştur veya alt başlık kalıntısıdır, onu atla
    df_secilen = df_secilen.iloc[1:].copy() 
    
    # Havalimanı adlarını temizle
    df_secilen[havalimani_sutunu] = df_secilen[havalimani_sutunu].astype(str).str.strip()
    
    # Sayısal veriyi temizle
    df_secilen[hedef_dis_hat_sutunu] = pd.to_numeric(
        df_secilen[hedef_dis_hat_sutunu].astype(str)
        .str.replace('.', '', regex=False).str.replace(',', '', regex=False),
        errors='coerce'
    ).fillna(0).astype(int)

    # Toplam, Not vb. satırları filtrele
    df_sonuc = df_secilen[
        ~df_secilen[havalimani_sutunu].astype(str).str.contains(
            "TOPLAMI|GENELİ|TRANSİT|\(|\*|\*\*|nan|Havalimanları|YOLCU TRAFİĞİ|Başkanlığı|işletilmekte",
            na=False, regex=True, case=False
        ) &
        df_secilen[havalimani_sutunu].notna() &
        (df_secilen[havalimani_sutunu] != '')
    ].copy()
    df_sonuc.reset_index(drop=True, inplace=True)
    print("-> Veri temizlendi ve gereksiz satırlar filtrelendi.")

    # --- Sonucu Göster ---
    print(f"\n--- SONUÇ: {hedef_yil} Yılı Dış Hat Verisi (İlk 10 Satır) ---")
    print(df_sonuc.head(10).to_markdown(index=False, floatfmt=","))

except FileNotFoundError:
    print(f"HATA: Dosya bulunamadı - {dosya_yolu}")
except Exception as e:
    print(f"HATA: İşlem sırasında beklenmedik hata: {e}")
    # Detaylı hata için:
    # import traceback
    # traceback.print_exc()