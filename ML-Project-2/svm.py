# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:57:52 2024

@author: Alican
"""

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import zero_one_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


def train_svm(X_train, y_train, kernel, C, **kwargs):
    """SVM modelini eğit."""
    model = SVC(kernel=kernel, C=C,**kwargs)
    model.fit(X_train, y_train)
    return model

def get_C_values():
    """C değerlerini belirler: 0.1'den 1'e kadar 0.1 artarak, sonra 1'den 50'ye kadar 1 artarak ve sonra 50'den 1000'e kadar 50 artarak."""
    C_values_1 = np.arange(0.1, 1.1, 0.1)    # 0.1'den 1'e kadar 0.1 artarak
    C_values_2 = np.arange(1, 51, 1)         # 1'den 50'ye kadar 1 artarak
    C_values_3 = np.arange(50, 1001, 50)     # 50'den 1000'e kadar 50 artarak
    return np.concatenate([C_values_1, C_values_2, C_values_3])


def train_with_params(X_train, y_train, X_val, y_val, kernel, C_values=None, **kwargs):
    """
    Farklı hiperparametrelerle modeli eğiterek doğruluk/loss değerlerini takip et.

    Parametreler:
    - X_train, y_train: Eğitim veri seti.
    - X_val, y_val: Doğrulama veri seti.
    - kernel: SVM çekirdek tipi.
    - C_values: Denenecek C değerleri listesi.

    Çıktı:
    - Eğitim ve doğrulama doğruluk/loss değerleri listeleri.
    """
    
    if C_values is None:
        C_values = get_C_values()

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for C in C_values:
        model = train_svm(X_train, y_train, kernel, C, probability=True)
        
        # Doğruluk hesaplaması
        train_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
        val_accuracies.append(accuracy_score(y_val, model.predict(X_val)))
        
        # Loss hesaplaması
        train_losses.append(log_loss(y_train, model.predict_proba(X_train)))
        val_losses.append(log_loss(y_val, model.predict_proba(X_val)))

    return train_accuracies, val_accuracies, train_losses, val_losses, C_values

def plot_train_wp(train_accs, val_accs, train_losses, val_losses, C_vals):
    """
    Eğitim ve doğrulama doğruluklarını ve kayıplarını grafikte çizer ve en iyi C değerini işaretler.

    Parametreler:
    - train_accs: Eğitim doğrulukları.
    - val_accs: Doğrulama doğrulukları.
    - train_losses: Eğitim kayıpları.
    - val_losses: Doğrulama kayıpları.
    - C_vals: C değerleri listesi.
    - best_C: En iyi doğrulama kaybı veren C değeri.
    """
    # Doğruluk grafiği
    plt.plot(C_vals, train_accs, label="Train Accuracy", marker='o')
    plt.plot(C_vals, val_accs, label="Validation Accuracy", marker='o')
    plt.xscale('log')
    plt.xlabel("C Value (Log Scale)")
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy for Different C Values")
    plt.legend()

    # Kayıp grafiği
    plt.figure()
    plt.plot(C_vals, train_losses, label="Train Loss", marker='o')
    plt.plot(C_vals, val_losses, label="Validation Loss", marker='o')
    plt.xscale('log')
    plt.xlabel("C Value (Log Scale)")
    plt.ylabel("Log Loss")
    plt.title("SVM Loss for Different C Values")
    plt.legend()

    plt.show()
    
    return

def evaluate_performance(model, X_train, X_val, X_test, y_train, y_val, y_test,best_C):
    """
    En başarılı model için doğruluk, precision, recall, f1 score ve confusion matrix hesapla.
    
    Parametreler:
    - model: Eğitimli SVM modelini.
    - X_train, X_val, X_test: Eğitim, doğrulama ve test verileri.
    - y_train, y_val, y_test: Eğitim, doğrulama ve test etiketleri.

    Çıktılar:
    - Performans metrikleri ve confusion matrix.
    """
    # Tahminler
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

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
    
    """
    # Sonuçları yazdırma
    print("Eğitim Seti - Doğruluk: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
        accuracy_train, precision_train, recall_train, f1_train))
    print("Doğrulama Seti - Doğruluk: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
        accuracy_val, precision_val, recall_val, f1_val))
    print("Test Seti - Doğruluk: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
        accuracy_test, precision_test, recall_test, f1_test))
    """
    
    results = f"""
    f"{model}"
    En İyi C Değeri: {best_C}
    Eğitim Seti - Doğruluk: {accuracy_train:.2f}, Precision: {precision_train:.2f}, Recall: {recall_train:.2f}, F1-Score: {f1_train:.2f}
    Doğrulama Seti - Doğruluk: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1-Score: {f1_val:.2f}
    Test Seti - Doğruluk: {accuracy_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1-Score: {f1_test:.2f}
    """
    
    # Confusion Matrix eklemek isterseniz:
    results += f"""
    Confusion Matrix - Eğitim Seti:
    {cm_train}

    Confusion Matrix - Doğrulama Seti:
    {cm_val}

    Confusion Matrix - Test Seti:
    {cm_test}
    """
    print(results)
      
    # Confusion Matrix Görselleştirme
    def plot_confusion_matrix(cm, title):
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            plt.title(title)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()
    
    plot_confusion_matrix(cm_train, "Eğitim Seti Confusion Matrix")
    plot_confusion_matrix(cm_val, "Doğrulama Seti Confusion Matrix")
    plot_confusion_matrix(cm_test, "Test Seti Confusion Matrix")
    
    return results

def save_results_to_txt(results, model_name, filename=None):
    """
    Model performans sonuçlarını bir .txt dosyasına kaydeder.

    Parametreler:
    - results (str): Dosyaya yazılacak sonuçların metni.
    - model_name (str): Modelin ismi veya tipi (örn. 'svm_linear').
    - filename (str, opsiyonel): Kaydedilecek dosyanın ismi. Eğer sağlanmazsa model adı kullanılacak.
    """
    if filename is None:
        # Modelin adını kullanarak dosya ismi oluşturuyoruz
        filename = f"{model_name}_performance.txt"
    
    try:
        # Dosyaya yazma işlemi
        with open(filename, 'w') as file:
            file.write(results)
        print(f"Sonuçlar '{filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Dosyaya yazarken bir hata oluştu: {e}")
    return


def plot_decision_boundary(model, X, y):
    """Karar sınırını çiz."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.title("Karar Sınırı")
    plt.show()
    return

