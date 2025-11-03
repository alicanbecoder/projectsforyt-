# 🇹🇷 Turizm Verileri ile Dış Hat Yolcu Sayısı Tahmini (2015–2025)

Bu proje, Türkiye’nin 2015–2025 dönemine ait **turizm istatistiklerini**, **makroekonomik göstergeleri** ve **pandemi etkilerini** bir araya getirerek dış hat yolcu sayısı ve turizm gelirlerinin **makine öğrenmesi yöntemleriyle tahmin edilmesini** amaçlamaktadır.  

## 🧩 Proje Özeti
- Veriler TÜİK, TCMB EVDS ve İTOSAM kaynaklarından elde edilmiştir.  
- Tüm veri setleri Python kullanılarak **otomatik biçimde okunmuş, birleştirilmiş ve temizlenmiştir.**
- Modellemelerde **Random Forest** ve **XGBoost** algoritmaları kullanılmıştır.
- 2025 yılı için **iyimser, nötr ve kötümser senaryolar** altında tahminler üretilmiştir.
- Görselleştirme adımları **Tableau** üzerinden yapılmıştır.

## 🧠 Kullanılan Yöntemler
- **Makine Öğrenmesi:** Random Forest Regressor, XGBoost Regressor  
- **Özellik Mühendisliği:** Mevsimsellik, pandemi değişkeni, post-COVID toparlanma, yıllık değişim oranları  
- **Senaryo Analizi:** 2025 yılı için döviz, TÜFE ve mevsimsel varyasyonlara göre üç farklı senaryo (iyimser / nötr / kötümser)

## 📈 Sonuçlar
| Dönem / Senaryo | Toplam Yolcu Sayısı | 2024’e Göre Değişim (%) |
|------------------|----------------------|---------------------------|
| **2024 Tahmini** | **137.529.516** | — |
| **2025 İyimser** | **145.376.967** | **+5.71%** |
| **2025 Nötr** | **138.454.254** | **+0.67%** |
| **2025 Kötümser** | **131.531.542** | **−4.36%** |

## 🗂 Veri Kaynakları
- **TCMB EVDS:** Döviz kurları (USD/TRY, EUR/TRY)  
- **TÜİK:** Turizm gelirleri ve yolcu istatistikleri  
- **İTOSAM:** Otel doluluk oranı ve şehir bazlı turizm göstergeleri  

Tüm veriler açık kaynaklı olup, bu proje kapsamında **Python kullanılarak harmanlanmış ve dönüştürülmüştür.**  
Veri setleri doğrudan CSV veya Excel olarak çekilip birleştirilmiştir.**

## 📊 Görselleştirme
- Tableau Sheets
- İçerik:
  - Havalanı bazlı dış hat yolcu sayısı
  - Yıllık yolcu sayısı değişimi
  - Ülke bazlı analizler
  - Döviz – gelir ilişkisi
  - Trend analizi

## 👨‍💻 Geliştirici
**Alican Tunç**  
💡 “Veriyle hikaye anlatmak, öngörüleri görünür kılmaktır.”  
