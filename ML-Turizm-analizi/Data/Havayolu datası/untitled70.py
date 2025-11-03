# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 03:05:29 2025

@author: Alican
"""

import pandas as pd
import numpy as np
import glob # Dosyaları toplu bulmak için
import os   # Klasör işlemleri için
import re   # Yıl/Ay bilgisi için

# --- Adım 1: Hedef Havalimanı Listesi ---
hedef_havalimanlari_raw = [
    "İstanbul Atatürk",
    "İstanbul(*)",
    "İstanbul Sabiha Gökçen(*)",
    "Ankara Esenboğa",
    "İzmir Adnan Menderes",
    "Antalya",
    "Gazipaşa Alanya(*)",
    "Muğla Dalaman",
    "Muğla Milas-Bodrum",
    "Adana",
    "Trabzon",
    "Erzurum",
    "Gaziantep",
    "Adıyaman",
    "Ağrı Ahmed-i Hani",
    "Amasya Merzifon",
    "Aydın Çıldır(*)",
    "Balıkesir Koca Seyit",
    "Balıkesir Merkez",
    "Batman",
    "Bingöl",
    "Bursa Yenişehir",
    "Çanakkale",
    "Çanakkale Gökçeada",
    "Denizli Çardak",
    "Diyarbakır",
    "Elazığ",
    "Erzincan Yıldırım Akbulut",
    "Eskişehir Hasan Polatkan(*)",
    "Hakkari Yüksekova Selahaddin Eyyubi",
    "Hatay",
    "Iğdır Şehit Bülent Aydın",
    "Isparta Süleyman Demirel",
    "Kahramanmaraş",
    "Kars Harakani",
    "Kastamonu",
    "Kayseri",
    "Kocaeli Cengiz Topel",
    "Konya",
    "Malatya",
    "Mardin Prof. Dr. Aziz Sancar",
    "Muş Sultan Alparslan",
    "Kapadokya", 
    "Ordu-Giresun",
    "Rize-Artvin",
    "Samsun Çarşamba",
    "Siirt",
    "Sinop",
    "Sivas Nuri Demirağ",
    "Şanlıurfa GAP", 
    "Şırnak Şerafettin Elçi",
    "Tekirdağ Çorlu Atatürk",
    "Tokat",
    "Uşak",
    "Van Ferit Melen",
    "Zafer(*)",
    "Zonguldak Çaycuma(*)"
]
hedef_havalimanlari_temiz = [h.strip() for h in hedef_havalimanlari_raw]
print(f"-> Filtreleme için kullanılacak havalimanı sayısı: {len(hedef_havalimanlari_temiz)}")

# --- Fonksiyon Tanımı (Öncekiyle Aynı) ---
def dhmi_12_ay_birlestir_filtreli(klasor_yolu, dosya_kalibi="YOLCU_*.xlsx", hedef_liste=None):
    """
    Belirtilen klasördeki 12 aylık DHMİ Excel dosyalarını okur. Her dosyadan 
    Havalimanları (1. sütun) ve Dış Hat kümülatif verilerini (Aylık dosyalarda 3., 
    Aralık'ta 6. sütun) alır. SADECE hedef_liste'deki havalimanlarını filtreler, 
    temizler ve 12 ayı birleştirip kümülatif DataFrame döndürür.
    """
    if hedef_liste is None:
        print("HATA: Hedef havalimanı listesi (hedef_liste) belirtilmedi.")
        return None
        
    print(f"\nKlasör taranıyor: {klasor_yolu}, Desen: {dosya_kalibi}")

    # 1. Dosyaları Bul ve Sırala
    try:
        tam_dosya_kalibi = os.path.join(klasor_yolu, dosya_kalibi)
        dosya_listesi = glob.glob(tam_dosya_kalibi)
        dosya_listesi.sort() 

        if not dosya_listesi:
            print(f"-> UYARI: '{dosya_kalibi}' desenine uygun dosya bulunamadı. Bu yıl atlanıyor.")
            return None # Hata değil, sadece veri yok
        print(f"Bulunan dosyalar: {len(dosya_listesi)} adet.")
        if len(dosya_listesi) != 12:
             print(f"-> UYARI: Beklenen 12 dosya yerine {len(dosya_listesi)} dosya bulundu!")

    except Exception as e:
        print(f"HATA: Dosyalar aranırken hata oluştu: {e}")
        return None

    # 2. Her Dosyayı Oku, Filtrele, Seç, Temizle
    tum_aylarin_filtrelenmis_verisi = []
    
    for dosya_yolu in dosya_listesi:
        dosya_adi = os.path.basename(dosya_yolu)
        try:
            # --- Ay Numarasını Al ---
            match = re.search(r'_(\d{2})\.', dosya_adi) 
            if match:
                ay_numarasi = int(match.group(1))
            else:
                 print(f"-> Uyarı: '{dosya_adi}' adında ay numarası bulunamadı. Atlanıyor.")
                 continue

            # --- Ana Veriyi Oku ---
            df = pd.read_excel(dosya_yolu, header=1, skiprows=[0, 2])

            # --- Dış Hat Sütununu Seç ---
            dis_hat_index = 2 if ay_numarasi != 12 else 5 

            if df.shape[1] <= dis_hat_index:
                 print(f"-> HATA: '{dosya_adi}' dosyasında beklenen Dış Hat sütunu (index {dis_hat_index}) yok. Atlanıyor.")
                 continue

            # --- Sütunları Seç (Ham hali) ---
            havalimani_sutunu = df.columns[0] 
            dis_hat_sutunu_adi_orj = df.columns[dis_hat_index] 
            df_secilen_ham = df[[havalimani_sutunu, dis_hat_sutunu_adi_orj]].copy()
            df_secilen_ham.columns = ['Havalimanları_Ham', f'Dış_Hat_Ham_{ay_numarasi:02d}']

            # --- Havalimanı Adını Temizle (Filtreleme için) ---
            df_secilen_ham['Havalimanları'] = df_secilen_ham['Havalimanları_Ham'].astype(str).str.strip().str.replace(" (*)","(*)", regex=False)

            # --- Hedef Listeye Göre Filtrele ---
            df_filtrelenmis = df_secilen_ham[
                df_secilen_ham['Havalimanları'].isin(hedef_liste)
            ].copy()

            if df_filtrelenmis.empty:
                continue 
                
            # --- Filtrelenmiş Veriyi Temizle ve Adlandır ---
            kumulatif_sutun_adi = f"Kumulatif_{ay_numarasi:02d}" 
            df_sonuc_tek_ay = pd.DataFrame() 
            df_sonuc_tek_ay['Havalimanları'] = df_filtrelenmis['Havalimanları'] 
            
            df_sonuc_tek_ay[kumulatif_sutun_adi] = pd.to_numeric(
                df_filtrelenmis[f'Dış_Hat_Ham_{ay_numarasi:02d}'].astype(str) 
                .str.replace('.', '', regex=False).str.replace(',', '', regex=False),
                errors='coerce' 
            ).fillna(0).astype(int) 

            df_sonuc_tek_ay.set_index('Havalimanları', inplace=True) 
            
            tum_aylarin_filtrelenmis_verisi.append(df_sonuc_tek_ay)
            # print(f"-> Başarılı: {kumulatif_sutun_adi} verisi {len(df_sonuc_tek_ay)} havalimanı için çekildi.") # Daha az çıktı

        except Exception as e:
            print(f"HATA: '{dosya_adi}' işlenirken hata oluştu: {e}")

    # 3. Tüm Filtrelenmiş Ayları Birleştir
    if not tum_aylarin_filtrelenmis_verisi:
        print("-> UYARI: Bu yıl için işlenecek geçerli veri bulunamadı.")
        return None

    try:
        df_birlesik_kumulatif = pd.DataFrame(index=hedef_liste) 
        
        for df_ in tum_aylarin_filtrelenmis_verisi:
            df_birlesik_kumulatif = df_birlesik_kumulatif.merge(df_, left_index=True, right_index=True, how='left')

        df_birlesik_kumulatif = df_birlesik_kumulatif.fillna(0).astype(int)
        df_birlesik_kumulatif = df_birlesik_kumulatif[sorted(df_birlesik_kumulatif.columns)]
        df_birlesik_kumulatif.index.name = 'Havalimanları'

        print("-> Filtrelenmiş tüm ayların kümülatif verileri başarıyla birleştirildi.")
        return df_birlesik_kumulatif

    except Exception as e:
        print(f"HATA: Kümülatif veriler birleştirilirken hata oluştu: {e}")
        return None

# --- YILLAR ARASI OTOMASYON DÖNGÜSÜ ---

# 1. Ana Klasör Yolunu Tanımla (Yıl klasörlerinin bir üstü)
base_klasor_yolu = r"C:\Users\Alican\Desktop\Çalışma grubu\Havayolu datası"

# 2. İşlenecek Yıl Aralığını Belirt
baslangic_yili = 2015
bitis_yili = 2024 # Dahil

# 3. Yıllar İçin Döngü Başlat
for yil in range(baslangic_yili, bitis_yili + 1):
    yil_str = str(yil)
    print(f"\n{'='*10} İŞLENEN YIL: {yil_str} {'='*10}")

    # O yıla ait klasör yolunu ve dosya kalıbını oluştur
    # DİKKAT: Klasör yapına göre "pdf24_convertPdfTo (1)" kısmını ayarla
    excel_klasor_yolu_yil = os.path.join(base_klasor_yolu, yil_str, "pdf24_convertPdfTo (1)") 
    aranacak_kalip_yil = f"YOLCU_{yil_str}_*.xlsx" 

    # Kümülatif Veriyi Oluşturmak İçin Fonksiyonu Çağır
    kumulatif_filtrelenmis_df = dhmi_12_ay_birlestir_filtreli(
        excel_klasor_yolu_yil, 
        aranacak_kalip_yil, 
        hedef_liste=hedef_havalimanlari_temiz 
    )

    # Kümülatif Veri Başarıyla Oluşturulduysa Aylık Veriyi Hesapla
    if kumulatif_filtrelenmis_df is not None:
        print(f"\n{yil_str} yılı kümülatif veriden aylık veriyi hesaplama...")
        try:
            df_aylik = kumulatif_filtrelenmis_df.copy()
            sirali_sutunlar = sorted(df_aylik.columns)
            
            # Eğer o yıl için 12 ay verisi tam değilse uyarı ver
            if len(sirali_sutunlar) < 12:
                 print(f"-> UYARI: {yil_str} yılı için {len(sirali_sutunlar)} aylık kümülatif veri bulundu. Aylık hesaplama eksik olabilir.")

            df_aylik = df_aylik[sirali_sutunlar] # Sıralı sütunları al
            
            df_aylik_fark = df_aylik.diff(axis=1)
            
            df_sonuc_aylik = pd.DataFrame(index=df_aylik.index)
            
            if sirali_sutunlar: # Eğer en az 1 sütun varsa
                ilk_ay_sutunu = sirali_sutunlar[0] 
                df_sonuc_aylik[f"Ay_{ilk_ay_sutunu.split('_')[1]}"] = df_aylik[ilk_ay_sutunu]
                # print(f"-> {ilk_ay_sutunu} (Ocak) verisi direkt alındı.") # Daha az çıktı
                
                for i in range(1, len(sirali_sutunlar)):
                    mevcut_sutun = sirali_sutunlar[i] 
                    yeni_sutun_adi = f"Ay_{mevcut_sutun.split('_')[1]}" 
                    df_sonuc_aylik[yeni_sutun_adi] = df_aylik_fark[mevcut_sutun]
                    # print(f"-> {yeni_sutun_adi} hesaplandı.") # Daha az çıktı
            
            df_sonuc_aylik = df_sonuc_aylik.fillna(0).astype(int)
            
            print(f"-> {yil_str} yılı aylık verileri başarıyla hesaplandı.")

            # --- Sonucu Kaydet (Aylık) ---
            cikti_dosya_adi = f"SONUC_Aylik_Dis_Hat_{yil_str}.xlsx"
            # Çıktıyı ilgili yılın klasörüne kaydet
            cikti_dosya_yolu_aylik = os.path.join(excel_klasor_yolu_yil, cikti_dosya_adi) 
            try:
                df_sonuc_aylik.to_excel(cikti_dosya_yolu_aylik, index=True) 
                print(f"-> Aylık sonuç başarıyla kaydedildi:\n   '{cikti_dosya_yolu_aylik}'")
            except Exception as e:
                print(f"-> HATA: Aylık sonuç dosyası kaydedilemedi: {e}")

        except Exception as e:
            print(f"HATA: {yil_str} yılı aylık verileri hesaplanırken hata oluştu: {e}")

    else:
        print(f"\n{yil_str} yılı için kümülatif veri oluşturulamadı, aylık hesaplama atlanıyor.")

print(f"\n{'='*10} OTOMASYON TAMAMLANDI ({baslangic_yili}-{bitis_yili}) {'='*10}")