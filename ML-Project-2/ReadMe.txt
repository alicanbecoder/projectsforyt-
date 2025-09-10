Açıklama
Bu proje, ikili sınıflandırma problemleri için yapay sinir ağları (YSA) ve destek vektör makineleri (DVM) modellerinin performanslarını karşılaştırmayı amaçlamaktadır. Proje, TensorFlow, Keras ve Scikit-learn kütüphaneleri kullanılarak Python ortamında geliştirilmiştir.

Gereksinimler
Projeyi çalıştırmak için gerekli Python kütüphanelerini yüklemek üzere aşağıdaki komutu kullanabilirsiniz:

pip install -r requirements.txt


Projeyi çalıştırmak için aşağıdaki adımları izleyin:

Veri Setini Oluşturma ve Görselleştirme:
Veri setini oluşturmak ve görselleştirmek için aşağıdaki komutu kullanın:

bash
python dataset.py
Yapay Sinir Ağları (YSA) Modellerini Eğitme ve Değerlendirme:
YSA modellerini eğitmek ve performanslarını değerlendirmek için aşağıdaki komutu kullanın:

bash
python neural_network.py
Destek Vektör Makineleri (DVM) Modellerini Eğitme ve Değerlendirme:
DVM modellerini eğitmek ve performanslarını değerlendirmek için aşağıdaki komutu kullanın:

bash
python svm.py

Ana Programı Çalıştırma:
Tüm modelleri eğitmek ve sonuçları görselleştirmek için ana programı çalıştırın:

bash
python main.py
Dosya Düzeni

Proje klasörü aşağıdaki şekilde düzenlenmiştir:


proje_klasoru/
 |- main.py                # Ana program, tüm modelleri eğitir ve sonuçları görselleştirir
 |- neural_network.py      # Yapay sinir ağları modellerini eğitir ve değerlendirir
 |- svm.py                 # Destek vektör makineleri modellerini eğitir ve değerlendirir
 |- dataset.py             # Veri setini oluşturur ve görselleştirir
 |- requirements.txt       # Gerekli Python kütüphanelerinin listesi
 |- results/               # Eğitim ve değerlendirme sonuçlarının kaydedildiği klasör
Detaylar
main.py: Ana program, tüm modelleri eğitir ve sonuçları görselleştirir.

neural_network.py: Yapay sinir ağları modellerini eğitir ve performanslarını değerlendirir.

svm.py: Destek vektör makineleri modellerini eğitir ve performanslarını değerlendirir.

dataset.py: Veri setini oluşturur ve görselleştirir.

requirements.txt: Projenin çalışması için gerekli Python kütüphanelerini içerir.

results/: Eğitim ve değerlendirme sonuçlarının (örneğin, kayıp grafikleri, doğruluk metrikleri) kaydedildiği klasördür.

Notlar
Proje, Python 3.x ile uyumludur.

Eğitim ve değerlendirme işlemleri sırasında oluşturulan grafikler ve metrikler results/ klasörüne kaydedilir.

Veri seti, Scikit-learn'in make_moons fonksiyonu kullanılarak oluşturulmuştur ve proje içerisinde otomatik olarak oluşturulur.

Eğer diğer not dizisi açılmazsa lütfen birlikte çalıştır ile .txt ile açınız.