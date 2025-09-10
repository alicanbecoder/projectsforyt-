# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import train_test_split


def normalize_data(df):
        """Belirli bir data frameyi min ve max yöntemine göre normalize eder.
        Argümanlar:
        df-data seti
        """
        normalized_data = (df - np.min(df)) / (np.max(df) - np.min(df))
        return (normalized_data)

def split_data(df):
    """ Data frameyi hesaplamar için ayırmak ve Eğitim, Doğrulama ve Test Kümesini
    oluşturmak için kullanılır.
    Argümanlar:
    df-data seti
    """
    #1.veriyi sütün başlıklarına göre ayırma
    x1=df["ex1"]
    x2=df["ex2"]
    ytarget=df["res"]
    
    #Train etmek için veriyi normalize etme ve ayırmadan önce x1 ile x2 sütünlarını birleştirme
    X = np.column_stack((x1, x2))
    X = normalize_data(X)
    x1normal=X[:,0]
    x2normal=X[:,1]
    # 2. Veriyi Eğitim, Doğrulama ve Test Kümesine Ayırma
    # İlk %60 Eğitim, kalan %40'ı geçici veri
    Xtrain, Xtemp, ytrain, ytemp = train_test_split(X, ytarget, test_size=0.4, random_state=42, stratify=ytarget)

    # Geçici veriyi %50-%50 doğrulama ve test olarak bölme
    Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, test_size=0.5, random_state=42, stratify=ytemp)
    
    # Veriyi numpye ve fonksiyonlara uygun olacak şekilde kullanmak için bölme
    x1train=Xtrain[:,0]; x2train=Xtrain[:,1]
    x1val=Xval[:,0]; x2val=Xval[:,1]
    x1test=Xtest[:,0]; x2test=Xtest[:,1]
    y1train=ytrain.to_numpy();y1val=yval.to_numpy();y1test=ytest.to_numpy()
    
    return (x1,x2,ytarget,x1train,x1val,x1test,x2train,x2val,x2test,y1train,y1val,y1test,x1normal,x2normal)

def sigmoid(preactivation):
    """İçine yazılın değişikine sigmoid fonksiyonuna uyarlamak için kullanınan aktivitas
    yon fonksiyonu.
    Argümanlar:
    preactivation-fonksiyonun girdi değeri
    """
    activation = 1/(1+np.exp(-preactivation));
    return (activation)

def binary_cross_entropy_loss(yprediction,ytarget):
    """Cross entropi loss hesabı yapan fonksiyon.(Dipnot:log 0 ve 1 gibi değerlerden
    kaçmak için küçük bir gürültüsü eklenmiştir.)
    Argümanlar:
    yprediction-model tarafından tahmin edilen çıktısı
    ytarget- verinin gerçek çıktısı
    """
    epsilon = 1e-8
    loss = -(ytarget * np.log(yprediction + epsilon) + (1 - ytarget) * np.log(1 - yprediction + epsilon ))
    return(loss)

def logistic_res(x1,x2,w1,w2,w0):
    """Logistic regresyon tipini uyguladığımız fonksiyon.
    Argümanlar:
    x1-1.değişken girdisi
    x2-2.değişken girdisi
    w1-x1e bağlı ağırlık
    w2-x2e bağlı ağırlık
    w0-bias değeri
    """
    ymodel=w1*x1+w2*x2+w0
    return(ymodel)


def prediction(ymodel):
    """Logistic regresyon sonucu üretilen değerler tahmin çıktısı verilen 
    0.5 değerine göre dönüştürme işlemi
    Argümanlar:
    ymodel-Oluşturulan logistic regresyon modeli
    """
    i=0
    for i in range(0,len(ymodel)):
        if ymodel[i] >= 0.5:
            ymodel[i] = 1
        else:
            ymodel[i] = 0
    return(ymodel)

def model(x1, x2, w1, w2, w0):
    """Logistic model oluşturma fonksiyonu
    Argümanlar:
    x1-x2=1.ve 2. değişkenlerin girdisi
    w1-w2-w0= ağırlıklar ve bias
    """
    return(prediction(sigmoid(logistic_res(x1, x2, w1, w2, w0))))
    
def train(x1,x2,ytarget,lr,epochs,w1,w2,w0):
    """Belirli bir kümede modeli eğitmek için kullanılan fonksiyon.
    Argümanlar:
    x1-1.değişken girdisi
    x2-2.değişken girdisi
    ytarget-1 ve 2. değişkene bağlı olarak gerçek çıktı
    lr-öğrenme katsayısı
    epochs-epochs sayısı
    w1,w2,w0-ağırlıklar ve bias değeri
    """
    l=len(ytarget)
    epoch_lossgd=[]
    weights=[]
    for epoch in range (epochs):
        epoch_loss=[]
        for i in range(l):# Her veri örneği için
            n=[]
            xi1 = x1[i]
            xi2 = x2[i]
            yi = ytarget[i]
            z = logistic_res(xi1, xi2, w1, w2, w0)
            yhat = sigmoid(z)
            # Cross-entropy kaybı
            loss = binary_cross_entropy_loss(yhat, yi)
            epoch_loss.append(loss)
            # Gradyanları hesaplama
            dw1 = (yhat - yi) * xi1
            dw2 = (yhat - yi) * xi2
            dw0 = (yhat - yi)
            
            # Parametreleri güncelleme
            w1 = w1 - lr * dw1
            w2 = w2 - lr * dw2
            w0 = w0 - lr * dw0
        #Her epoch sonunda kaybı yazdırma
            n.append(epoch_loss)
        epoch_lossgd.append(np.mean(n))
        weights.append((w1, w2, w0))
        """
        if epoch % 100 == 0:
           print(f"Epoch {epoch}, Loss: {(epoch_lossgd[epoch]):.4f}")
        """
    return w1, w2, w0, epoch_lossgd, weights

def validloss(we,x1val,x2val,ylambdaval,epochs):
    """Doğrulama kümesinde loss değerini hesaplayan fonksiyon.
    Argümanlar:
    w11-Ağırlık ve bias vektörü
    x1val-1.değişkene bağlı doğrulama kümesi
    x2val-2.değişkene bağlı doğrulama kümesi
    ylambdaval-1 ve 2. değişkene bağlı olarak gerçek doğrulama çıktısı
    epochs-epoch değeri
    """
    losses=[]
    for i, w in enumerate(we[:epochs]):
        w1,w2,w0=w
        ypred=sigmoid(logistic_res(x1val, x2val, w1, w2, w0))
        loss = np.mean(binary_cross_entropy_loss(ypred,ylambdaval))
        losses.append(loss)
    return(losses)

def calculate_metrics(y_true, y_pred):
    """Accuracy, precision, recall, f1_score değerlerini hesaplamak için 
    kullanılan fonksiyon.
    Argümanlar:
    y_true-gerçek çıktılar
    y_pred-model çıktıları
    """
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FN = sum((y_true == 1) & (y_pred == 0))
    
    # Accuracy
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    # Precision
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0  # Division by zero durumunda 0 döndür
    
    # Recall
    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0  # Division by zero durumunda 0 döndür
    
    # F1-Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0  # Division by zero durumunda 0 döndür
        
    print(accuracy, precision, recall, f1_score)
    
    return accuracy, precision, recall, f1_score
    
    