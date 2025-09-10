# -*- coding: utf-8 -*-

import pythoneval as pye
import pythontrain as pyt
import pandas as pd

#Data okuma ve data frame oluşturma kısmı.
df = pd.read_csv("dataset/hw1Data.txt")

#Ekte verilen örneklerin ilk %60’ını eğitim, sonraki %20’sini doğrulama, 
#kalan %20’sini test için kullanma kısmı
x1,x2,ytarget,x1train,x1val,x1test,x2train,x2val,x2test,y1train,y1val,y1test,x1normal,x2normal=pyt.split_data(df)

#örneklerin iki sınıfa dağılımını görmek için x eksenini 1. sınav notu, y eksenini 2. sınav 
#notu için kullanarak ve iki sınıfa ait örnekleri iki farklı renk ile göstererek örnekleri çizdirme kısmı.
pye.sınıf_plot(x1,x2,ytarget)

#Model Eğitme ve grafiklerini çizdirme
epochs_num=1000;lr=0.01
w11, w21, w00 , losstrain, we= pyt.train(x1train, x2train, y1train, lr, epochs_num,0,0,0)

#Eğitime ait loss grafiği
pye.lossplot(losstrain,"Eğitim lossu","b")
#Eğitime ait loss grafiği
yvalsapka=pyt.model(x1val, x2val, w11, w21, w00)
lossval=pyt.validloss(we,x1val,x2val,yvalsapka,epochs_num)
pye.lossplot(lossval,"Doğrulama lossu","r")
#Doğrulama ve eğitimi karşılaştıran grafik
pye.valossvstrainloss(lossval,losstrain)

#accuracy, precision, recall, f1_score hesaplama
yegitimsapka=pyt.model(x1train, x2train, w11, w21, w00)
ydogsapka=pyt.model(x1val, x2val, w11, w21, w00)
ytestsapka=pyt.model(x1test, x2test, w11, w21, w00)

pyt.calculate_metrics(y1train, yegitimsapka)
pyt.calculate_metrics(y1val, ydogsapka)
pyt.calculate_metrics(y1test, ytestsapka)

#Sonuçlar ile sınıf dağılımı ve karar sınırı
pye.sınıf_plot2(x1normal, x2normal, ytarget, w11, w21, w00)




