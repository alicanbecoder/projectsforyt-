# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def sınıf_plot(x1,x2,y):
    """2 değişkenli bir sistem için sınıflara göre dağılımnı plot eden fonksiyon.
    Argümanlar:
    x1-1.değişken girdisi
    x2-2.değişken girdisi
    y-1 ve 2. değişkene bağlı olarak çıktı
    """
    class_0_x = [x1[i] for i in range(len(y)) if y[i] == 0]
    class_0_y = [x2[i] for i in range(len(y)) if y[i] == 0]

    class_1_x = [x1[i] for i in range(len(y)) if y[i] == 1]
    class_1_y = [x2[i] for i in range(len(y)) if y[i] == 1]

    plt.scatter(class_0_x, class_0_y, color='red', label='Sınıf 0')
    plt.scatter(class_1_x, class_1_y, color='blue', label='Sınıf 1')

    plt.xlabel('1. Sınav Notu')
    plt.ylabel('2. Sınav Notu')
    plt.title('Sınıf Dağılımı')

    plt.legend()

    plt.show()
    return()

def dagılım(ytarget):
    """ 0 ve 1 sınıflarının dağılımını plot eden fonksiyon
    Argümanlar:
    ytarget-Çıktı değerleri
    """
    class_counts = ytarget.value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(class_counts.index, class_counts.values, color=['blue', 'red'], alpha=0.7)
    plt.xticks([0, 1], labels=['0', '1'], fontsize=12)
    plt.ylabel('Frekans (Adet)', fontsize=12)
    plt.xlabel('Sınıflar', fontsize=12)
    plt.title('Sınıfların Dağılımı', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
    return()

def lossplot(loss,title,color):
    """ Belirli loss değerlerini grafiğe döken fonksiyon.
    Argümanlar:
    loss-Loss değerleri
    title-başlık
    color-renk değeri
    """
    x = range(len(loss))
    # Grafik çizimi
    plt.figure(figsize=(10, 6))
    plt.plot(x, loss, color)  # Çizgi ve noktalar
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    return()

def valossvstrainloss(lossval,losstrain):
    """Herhangi iki loss değeri karşılaştırmak için grafiğe döken fonksiyon.
    Burada eğitim ve doğrulama loss değerlerini karşılaştırmak için yazıldı.
    Argümanlar:
    lossvall-Doğrulama Loss değerleri
    losstrain-Eğitim loss değerleri
    """
    y = range(len(lossval))
    plt.plot(y, losstrain, color='blue', label='Eğitim loss')
    plt.plot(y, lossval, color='red', label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation loss')
    plt.legend()
    plt.show()
    return()

def sınıf_plot2(x1, x2, y, w1, w2, w0):
    """
    2 değişkenli bir sistem için sınıflara göre dağılımı ve ağırlık parametrelerine göre sınır çizgisini çizen fonksiyon.
    
    Argümanlar:
    x1 - 1. değişken girdisi
    x2 - 2. değişken girdisi
    y - 1 ve 2. değişkene bağlı olarak çıktı
    w1, w2, w0 - ağırlık parametreleri (karar sınırı için)
    """
    # Sınıf 0 ve Sınıf 1 örneklerini ayır
    class_0_x = [x1[i] for i in range(len(y)) if y[i] == 0]
    class_0_y = [x2[i] for i in range(len(y)) if y[i] == 0]

    class_1_x = [x1[i] for i in range(len(y)) if y[i] == 1]
    class_1_y = [x2[i] for i in range(len(y)) if y[i] == 1]

    # Sınıfları scatter plot ile gösterme
    plt.scatter(class_0_x, class_0_y, color='red', label='Sınıf 0')
    plt.scatter(class_1_x, class_1_y, color='blue', label='Sınıf 1')

    # Karar sınırı için çizgi ekleme
    # Karar sınırı: w1 * x1 + w2 * x2 + w0 = 0 → x2 = -(w1/w2) * x1 - (w0/w2)
    x_vals = np.linspace(min(x1), max(x1), 100)  # x1 ekseni boyunca çizgi için değerler
    y_vals = -(w1 * x_vals + w0) / w2      # x2 değerlerini hesapla
    plt.plot(x_vals, y_vals, color='green', label='Karar Sınırı')

    # Eksen ve başlıklar
    plt.xlabel('1. Sınav Notu')
    plt.ylabel('2. Sınav Notu')
    plt.title('Sınıf Dağılımı ve Karar Sınırı')

    plt.legend()
    plt.show()
    return


