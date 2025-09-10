 # -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:38:15 2024

@author: Alican
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Input
import tensorflow as tf
import inspect
import numpy as np





def build_model(input_dim, hidden_layers, activation='sigmoid'):
    """Modeli oluşturma fonksiyonu"""
    model = tf.keras.Sequential()
    
    # İlk katman olarak Input kullanmak
    model.add(Input(shape=(input_dim,)))  
    
    # Gizli katmanları ekleme
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
    
    # Çıktı katmanı
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, optimizer, loss, epochs, batch_size=None):
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return history


def loss_models(input_dim, hidden_layers, X_train, y_train, X_val, y_val, epochs):
    sgd_optimizer = SGD(learning_rate=0.01)
    model_sgd = build_model(input_dim, hidden_layers, activation='sigmoid')
    history_sgd = train_model(model_sgd, X_train, y_train, X_val, y_val, sgd_optimizer, BinaryCrossentropy(), epochs=epochs)
    bgd_optimizer = SGD(learning_rate=0.01)
    model_bgd = build_model(input_dim, hidden_layers, activation='sigmoid')
    history_bgd = train_model(model_bgd, X_train, y_train, X_val, y_val, bgd_optimizer, BinaryCrossentropy(), epochs=epochs, batch_size=X_train.shape[0])
    batch_size = 10
    sgd_optimizer2 = SGD(learning_rate=0.01)
    model_mbgd = build_model(input_dim, hidden_layers, activation='sigmoid')
    history_mbgd = train_model(model_mbgd, X_train, y_train, X_val, y_val, sgd_optimizer2, BinaryCrossentropy(), epochs=epochs, batch_size=batch_size)
    return(history_sgd,model_sgd,history_bgd,model_bgd,history_mbgd,model_mbgd)


def plot_loss(history_sgd,history_bgd,history_mbgd):
    
    """
    Eğitim ve doğrulama kaybını (loss) karşılaştıran grafik çizer.
    
    Args:
    - history_sgd: Keras modelinin `fit` metodundan dönen history nesnesi.
    """
    # Eğitim ve doğrulama kayıplarını al
    train_loss_sgd = history_sgd.history['loss']
    val_loss_sgd = history_sgd.history['val_loss']

    # Grafik oluştur
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss_sgd) + 1), train_loss_sgd, label="Eğitim Kaybı (Train Loss)", marker='o')
    plt.plot(range(1, len(val_loss_sgd) + 1), val_loss_sgd, label="Doğrulama Kaybı (Validation Loss)", marker='x')

    # Eksenler ve başlıklar
    plt.title("Eğitim ve Doğrulama Kaybı Karşılaştırması (SGD)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    train_loss_bgd = history_bgd.history['loss']
    val_loss_bgd = history_bgd.history['val_loss']

    # Gösterim
    plt.show()
    
    # Grafik oluştur
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss_bgd) + 1), train_loss_bgd, label="Eğitim Kaybı (Train Loss)", marker='o')
    plt.plot(range(1, len(val_loss_bgd) + 1), val_loss_bgd, label="Doğrulama Kaybı (Validation Loss)", marker='x')

    # Eksenler ve başlıklar
    plt.title("Eğitim ve Doğrulama Kaybı Karşılaştırması (BGD)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Gösterim
    plt.show()
    
    train_loss_mbgd = history_mbgd.history['loss']
    val_loss_mbgd = history_mbgd.history['val_loss']
    
    
    # Grafik oluştur
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss_mbgd) + 1), train_loss_mbgd, label="Eğitim Kaybı (Train Loss)", marker='o')
    plt.plot(range(1, len(val_loss_mbgd) + 1), val_loss_mbgd, label="Doğrulama Kaybı (Validation Loss)", marker='x')

    # Eksenler ve başlıklar
    plt.title("Eğitim ve Doğrulama Kaybı Karşılaştırması (MBGD)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Gösterim
    plt.show() 
    
    return

   
def evaluate_performance_ann(model, X_train, X_val, X_test, y_train, y_val, y_test,model_name):
    """Modelin performansını değerlendirir ve sonuçları döndürür."""
    
    # Tahminler (modelin çıktısı 0 ile 1 arasında sürekli değerler döndürür)
    y_pred_train_prob = model.predict(X_train)
    y_pred_val_prob = model.predict(X_val)
    y_pred_test_prob = model.predict(X_test)
    
    # Tahminleri binary (0 veya 1) sınıflara dönüştürme
    threshold = 0.5
    y_pred_train = (y_pred_train_prob > threshold).astype(int)
    y_pred_val = (y_pred_val_prob > threshold).astype(int)
    y_pred_test = (y_pred_test_prob > threshold).astype(int)

    # Performans Metrikleri
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    precision_train = precision_score(y_train, y_pred_train)
    precision_val = precision_score(y_val, y_pred_val)
    precision_test = precision_score(y_test, y_pred_test)

    recall_train = recall_score(y_train, y_pred_train)
    recall_val = recall_score(y_val, y_pred_val)
    recall_test = recall_score(y_test, y_pred_test)

    f1_train = f1_score(y_train, y_pred_train)
    f1_val = f1_score(y_val, y_pred_val)
    f1_test = f1_score(y_test, y_pred_test)

    # Confusion matrix hesaplama
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_val = confusion_matrix(y_val, y_pred_val)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    
    results = f"""
    Eğitim Seti - Doğruluk: {accuracy_train:.2f}, Precision: {precision_train:.2f}, Recall: {recall_train:.2f}, F1-Score: {f1_train:.2f}
    Doğrulama Seti - Doğruluk: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1-Score: {f1_val:.2f}
    Test Seti - Doğruluk: {accuracy_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1-Score: {f1_test:.2f}
    Confusion Matrix - Eğitim Seti:
    {cm_train}

    Confusion Matrix - Doğrulama Seti:
    {cm_val}

    Confusion Matrix - Test Seti:
    {cm_test}
    """
    
    # Sonuçları ekrana yazdırma
    print(results)
    
    save_results_to_txt(results, model_name, filename=None)

    return


def save_results_to_txt(results, model_name, filename=None):
    """
    Model performans sonuçlarını bir .txt dosyasına kaydeder.
    
    Parametreler:
    - results (str): Dosyaya yazılacak sonuçların metni.
    - model_name (str): Modelin ismi (dosya adı için).
    - filename (str): Kaydedilecek dosyanın ismi (varsayılan None, model ismi kullanılır).
    """
    if not filename:
        filename = f"{model_name}.txt"
    
    try:
        # Dosyaya yazma işlemi
        with open(filename, 'w') as file:
            file.write(results)
        print(f"Sonuçlar '{filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Dosyaya yazarken bir hata oluştu: {e}")
    return

def plot_decision_boundary_ann(model, X, y):
    """
    Verilen modelin karar sınırını çizdirir.
    
    Parametreler:
    - model: Eğitilmiş model.
    - X: Veri noktaları (özellikler).
    - y: Veri noktalarının etiketleri.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # 2. Modelin tahminlerini al
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 3. Karar sınırını çiz
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("Modelin Karar Sınırı")
    plt.xlabel("Özellik 1")
    plt.ylabel("Özellik 2")
    plt.show()






