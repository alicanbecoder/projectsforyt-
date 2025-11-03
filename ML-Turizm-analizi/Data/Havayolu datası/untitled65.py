# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:45:50 2025

@author: Alican
"""

import pandas as pd
import numpy as np
import glob # Dosyaları toplu bulmak için
import os   # Klasör işlemleri için
import re   # Yıl/Ay bilgisi için

# ... (dhmi_aylik_birlestirici_ve_ayirici fonksiyonunun tamamı buraya gelecek - önceki yanıttaki gibi) ...
def dhmi_aylik_birlestirici_ve_ayirici(klasor_yolu, dosya_kalibi="YOLCU_*.xlsx"):
    """
    Belirtilen klasördeki DHMİ Excel dosyalarını okur, Havalimanları ve ilgili
    yılın Dış Hat kümülatif verilerini alır, birleştirir ve aylık veriyi hesaplar.

    Args:
        klasor_yolu (str): Excel dosyalarının bulunduğu klasörün yolu.
        dosya_kalibi (str): Aranacak dosya adı deseni (örn: "YOLCU_2023_*.xlsx").

    Returns:
        pandas.DataFrame or None: Başarılı olursa aylık veriyi içeren DataFrame,
                                   hata olursa None döndürür.
    """
    print(f"Klasör taranıyor: {klasor_yolu}")
    print(f"Aranan dosya kalıbı: {dosya_kalibi}")

    # 1. Dosyaları Bul ve Sırala
    try:
        # glob.glob tam dosya yolunu verir
        tam_dosya_kalibi = os.path.join(klasor_yolu, dosya_kalibi)
        dosya_listesi = glob.glob(tam_dosya_kalibi)
        # Dosyaları isme göre sırala (bu _01, _02.. sıralamasını sağlamalı)
        dosya_listesi.sort() 

        if not dosya_listesi:
            print(f"HATA: Belirtilen klasörde '{dosya_kalibi}' desenine uygun dosya bulunamadı.")
            return None
        
        print(f"Bulunan dosyalar ({len(dosya_listesi)} adet):")
        for f in dosya_listesi[:5]: # İlk 5'ini göster
             print(f" - {os.path.basename(f)}")
        if len(dosya_listesi) > 5: print(" ...")

    except Exception as e:
        print(f"HATA: Dosyalar aranırken hata oluştu: {e}")
        return None

    # 2. Her Dosyayı Oku ve Kümülatif Veriyi Çek
    tum_kumulatif_veriler = []
    ay_listesi = [] # Sütun adları için (örn: Kumulatif_1, Kumulatif_2)

    for dosya_yolu in dosya_listesi:
        dosya_adi = os.path.basename(dosya_yolu)
        print(f"\nİşleniyor: {dosya_adi}")
        try:
            # --- Ay Numarasını ve Yılı Dosya Adından veya İçerikten Al ---
            ay_numarasi = None
            hedef_yil = None
            # Önce dosya adından ayıklamaya çalış (örn: YOLCU_2023_03.xlsx)
            match = re.search(r'(\d{4})_(\d{2})', dosya_adi)
            if match:
                hedef_yil = match.group(1)
                ay_numarasi = int(match.group(2))
                print(f"-> Dosya adından yıl: {hedef_yil}, Ay: {ay_numarasi} bulundu.")
            else:
                # Dosya adından bulamazsa içerikten yılı almayı dene (yedek plan)
                header_df = pd.read_excel(dosya_yolu, nrows=2, header=None)
                ikinci_yil_baslik_raw = str(header_df.iloc[0, 4]) 
                yil_eslesme = re.search(r'\d{4}', ikinci_yil_baslik_raw)
                if yil_eslesme:
                    hedef_yil = yil_eslesme.group(0)
                else: hedef_yil = "XXXX"
                # Ay numarasını dosya sırasına göre ata (riskli ama alternatifsiz)
                ay_numarasi = len(tum_kumulatif_veriler) + 1 
                print(f"-> Dosya adında ay yok, içerikten yıl: {hedef_yil}, Sıra: {ay_numarasi} kullanıldı.")


            # --- Ana Veriyi Oku (header=1, skiprows=[0, 2]) ---
            df = pd.read_excel(dosya_yolu, header=1, skiprows=[0, 2])

            # --- Sütunları Seç (0 ve 5) ---
            havalimani_sutunu = df.columns[0]
            # ÖNEMLİ DÜZELTME: Kümülatif verilerde 6. sütun (index 5) ikinci yıla ait Dış Hat'tır.
            # Ancak, eğer dosya SADECE o aya ait kümülatif veriyi içeriyorsa,
            # genellikle 3. sütun (index 2) Dış Hat verisidir. Bunu kontrol etmeliyiz.
            # Yıl sonu (Aralık) dosyası gibi çift yıl içeren dosyalarda 6. sütun (index 5) doğrudur.
            # Tek aylık kümülatif dosyalarda (Mart Sonu vb.) 3. sütun (index 2) Dış Hattır.
            # Bu ayrımı yapmak için sütun sayısına bakabiliriz veya başlığa.
            # Şimdilik, tek aylık kümülatif olduğunu varsayarak 3. sütunu (index 2) deneyelim.
            # Eğer Aralık dosyasıysa veya çift yıl varsa bu index değişmeli.
            # DAHA GÜVENLİ YÖNTEM: Sütun adına göre bulmak. İkinci satırdaki başlığa bakalım.
            
            dis_hat_index = -1
            header_row_1 = pd.read_excel(dosya_yolu, header=None, skiprows=1, nrows=1).iloc[0].astype(str)
            for idx, col_name in enumerate(header_row_1):
                 if 'Dış Hat' in col_name:
                      dis_hat_index = idx # İlk Dış Hat sütununu bul
                      break

            if dis_hat_index == -1:
                 print(f"HATA: '{dosya_adi}' dosyasında 'Dış Hat' başlığı bulunamadı.")
                 continue # Bu dosyayı atla

            dis_hat_sutunu = df.columns[dis_hat_index]
            print(f"-> 'Dış Hat' sütunu {dis_hat_index}. indekste bulundu: {dis_hat_sutunu}")

            df_secilen = df[[havalimani_sutunu, dis_hat_sutunu]].copy()


            # --- Sütunları Adlandır ve Temizle ---
            kumulatif_sutun_adi = f"Kumulatif_{ay_numarasi:02d}" # Örn: Kumulatif_01, Kumulatif_12
            df_secilen.rename(columns={
                havalimani_sutunu: 'Havalimanları',
                dis_hat_sutunu: kumulatif_sutun_adi
            }, inplace=True)

            df_secilen['Havalimanları'] = df_secilen['Havalimanları'].astype(str).str.strip()
            df_secilen[kumulatif_sutun_adi] = pd.to_numeric(
                df_secilen[kumulatif_sutun_adi].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False),
                errors='coerce'
            ).fillna(0).astype(int)

            # --- Gereksiz Satırları Filtrele ---
            df_sonuc_tek_ay = df_secilen[
                ~df_secilen['Havalimanları'].astype(str).str.contains(
                    "TOPLAMI|GENELİ|TRANSİT|\(|\*|\*\*|nan|Havalimanları", 
                    na=False, regex=True) & 
                df_secilen['Havalimanları'].notna() & 
                (df_secilen['Havalimanları'] != '') &
                 (~df_secilen['Havalimanları'].isin(['İstanbul Atatürk', 'Aydın Çıldır (*)', 'Balıkesir Merkez', 'Çanakkale Gökçeada','Uşak']))
            ].copy()
            
            # Havalimanını index yap
            df_sonuc_tek_ay.set_index('Havalimanları', inplace=True) 

            tum_kumulatif_veriler.append(df_sonuc_tek_ay)
            ay_listesi.append(kumulatif_sutun_adi)
            print(f"-> Başarılı: {kumulatif_sutun_adi} verisi çekildi.")

        except Exception as e:
            print(f"HATA: {dosya_adi} işlenirken hata oluştu: {e}")
            # Hatalı dosyayı atla ve devam et ya da işlemi durdur
            # return None # Hata durumunda durdurmak için

    # 3. Tüm Kümülatif Verileri Birleştir
    if not tum_kumulatif_veriler:
        print("HATA: Hiçbir dosya başarıyla işlenemedi.")
        return None

    try:
        # İlk DataFrame'i al, diğerlerini index üzerinden birleştir
        # Index'leri (Havalimanları) birleştirmeden önce kontrol et, farklılıklar olabilir
        all_indices = pd.Index([])
        for df_ in tum_kumulatif_veriler:
            all_indices = all_indices.union(df_.index)

        # Boş bir DataFrame oluştur ve tüm ayları buraya ekle
        df_kumulatif_tum_aylar = pd.DataFrame(index=all_indices)
        for df_ in tum_kumulatif_veriler:
            df_kumulatif_tum_aylar = df_kumulatif_tum_aylar.merge(df_, left_index=True, right_index=True, how='left')


        # Eksik verileri 0 ile doldur
        df_kumulatif_tum_aylar = df_kumulatif_tum_aylar.fillna(0).astype(int)
        print("\n-> Tüm ayların kümülatif verileri başarıyla birleştirildi.")
        print("Birleştirilmiş Kümülatif Veri (İlk 5 Satır):")
        print(df_kumulatif_tum_aylar.head().to_markdown(floatfmt=","))

    except Exception as e:
        print(f"HATA: Kümülatif veriler birleştirilirken hata oluştu: {e}")
        return None

    # 4. Kümülatif Veriden Aylık Veriyi Hesapla (De-kümülasyon)
    try:
        df_aylik = df_kumulatif_tum_aylar.copy()
        
        # diff() fonksiyonu bir önceki sütunu çıkarır.
        # axis=1 sütunlar boyunca işlem yapılacağını belirtir.
        # Sütunların doğru sırada olduğundan emin olmalıyız (Kumulatif_01, _02, ...)
        df_aylik_fark = df_aylik[sorted(df_aylik.columns)].diff(axis=1) 
        
        # İlk sütunu (Ocak) orijinal kümülatiften alalım.
        ilk_ay_sutunu = sorted(df_aylik.columns)[0] # Örn: Kumulatif_01
        df_aylik[ilk_ay_sutunu] = df_kumulatif_tum_aylar[ilk_ay_sutunu] 
        
        # Sonraki ayları (Şubat'tan itibaren) fark değerleriyle doldur.
        # Sütun adlarını da temizleyelim (Kumulatif_02 -> Ay_02)
        yeni_sutun_adlari = {}
        yeni_sutun_adlari[ilk_ay_sutunu] = f"Ay_{ilk_ay_sutunu.split('_')[1]}" # Ay_01

        sorted_cols = sorted(df_aylik.columns)
        for i in range(1, len(sorted_cols)):
            mevcut_sutun = sorted_cols[i] # Örn: Kumulatif_02
            # Fark DataFrame'inden doğru sütunu al
            df_aylik[mevcut_sutun] = df_aylik_fark[mevcut_sutun] 
            yeni_sutun_adlari[mevcut_sutun] = f"Ay_{mevcut_sutun.split('_')[1]}" # Ay_02
            
        df_aylik.rename(columns=yeni_sutun_adlari, inplace=True)
        # Sütunları ay sırasına göre yeniden sırala
        df_aylik = df_aylik[sorted(df_aylik.columns)]


        # NaN kontrolü ve tip dönüşümü
        df_aylik = df_aylik.fillna(0).astype(int)
        
        print("\n-> Aylık veriler başarıyla hesaplandı (De-kümülasyon yapıldı).")
        return df_aylik

    except Exception as e:
        print(f"HATA: Aylık veriler hesaplanırken (de-kümülasyon) hata oluştu: {e}")
        return None


# --- FONKSİYONU KULLAN ---

# 1. Excel Dosyalarının Bulunduğu Klasörün Yolunu Buraya Yazın
excel_klasor_yolu = r"C:\Users\Alican\Desktop\Çalışma grubu\Havayolu datası\2023\pdf24_convertPdfTo (1)" 

# 2. Aranacak Dosya Kalıbını Düzeltiyoruz
aranacak_kalip = "YOLCU_2023_*.xlsx" # Dosya adlarına uygun kalıp

# 3. Fonksiyonu Çağır
aylik_sonuc_df = dhmi_aylik_birlestirici_ve_ayirici(excel_klasor_yolu, aranacak_kalip)

# 4. Sonucu Göster veya Kaydet
if aylik_sonuc_df is not None:
    print("\n--- SONUÇ: Hesaplanmış Aylık Dış Hat Yolcu Sayıları (İlk 5 Satır) ---")
    print(aylik_sonuc_df.head().to_markdown(floatfmt=","))

    # İstersen sonucu yeni bir Excel dosyasına kaydet
    cikti_dosya_yolu = os.path.join(excel_klasor_yolu, "SONUC_Aylik_Dis_Hat_2023.xlsx")
    try:
        aylik_sonuc_df.to_excel(cikti_dosya_yolu)
        print(f"\n-> Sonuç başarıyla '{cikti_dosya_yolu}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"\n-> HATA: Sonuç dosyası kaydedilemedi: {e}")

else:
    print("\nİşlem başarısız oldu. Hata mesajlarını kontrol edin.")