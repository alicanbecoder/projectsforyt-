# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:36:08 2024

@author: Alican
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Modeli yükleme
model_path = 'C:/Users/Alican/Desktop/model30.keras'
model = tf.keras.models.load_model(model_path)

# Test verisini hazırlama
TEST_PATH = r"C:\Users\Alican\Desktop\Code\BLM5135-Derin öğrenme ve Yapay Zeka\Project\train\Output3\test"
IMAGE_SIZE = 224
BATCH_SIZE = 4

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(directory=TEST_PATH,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

# Tahminlerde bulunma
predicted_classes = np.argmax(model.predict(test_gen), axis=1)

# Gerçek etiketleri elde etme
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# Karışıklık matrisini hesaplama
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Sınıflandırma raporunu hesaplama
class_report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# Karışıklık matrisini görselleştirme
df_cm = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt="d", annot_kws={"size": 10}, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Karışıklık matrisini kaydetme
plt.show()

# Sınıflandırma raporunu yazdırma ve kaydetme
print(class_report)
with open('classification_report.txt', 'w') as f:
    f.write(class_report)

print("Confusion matrix and classification report have been saved.")