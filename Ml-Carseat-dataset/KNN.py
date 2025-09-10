# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:34:00 2025

@author: Alican
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Dosya yolunu belirtelim
file_path = "C:\\Users\\Alican\\Desktop\\Ödev\\ml-100k\\u.data"

# Veri setini yükleyelim
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(file_path, sep='\t', names=columns)

# Eğitim ve test setlerine ayıralım
train_data, test_data = train_test_split(data, test_size=0.2, random_state=100)

print(f"Eğitim seti boyutu: {train_data.shape}")
print(f"Test seti boyutu: {test_data.shape}")

# Kullanıcı-Öğe matrisi oluştur
user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Kullanıcılar arası cosine benzerlik matrisi
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def predict_user_based(user_id, item_id, k=5):
    # Kullanıcı ve öğe ID'sinin veri setinde bulunup bulunmadığını kontrol et
    if user_id not in user_similarity_df.index or item_id not in user_item_matrix.columns:
        return np.nan
    
    # Kullanıcının benzerlik skorlarını al
    user_similarities = user_similarity_df[user_id]
    
    # Kullanıcının benzerlik skorlarına göre sıralama
    similar_users = user_similarities.sort_values(ascending=False).index[1:k+1]
    
    # Benzer kullanıcıların öğe değerlendirmelerini al
    similar_users_ratings = user_item_matrix.loc[similar_users, item_id]
    
    # Tahmini hesapla
    prediction = np.dot(user_similarities[similar_users], similar_users_ratings) / np.sum(user_similarities[similar_users])
    return prediction

def calculate_rmse_user_based(k):
    test_data['prediction'] = test_data.apply(lambda x: predict_user_based(x['user_id'], x['item_id'], k), axis=1)
    valid_predictions = test_data['prediction'].notna()
    filtered_test_data = test_data[valid_predictions]
    rmse = np.sqrt(mean_squared_error(filtered_test_data['rating'], filtered_test_data['prediction']))
    return rmse

# Öğe-temelli KNN (Item-KNN) Algoritması
item_user_matrix = train_data.pivot(index='item_id', columns='user_id', values='rating').fillna(0)

# Öğeler arası cosine benzerlik matrisi
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

def predict_item_based(user_id, item_id, k=5):
    # Kullanıcı ve öğe ID'sinin veri setinde bulunup bulunmadığını kontrol et
    if item_id not in item_similarity_df.index or user_id not in item_user_matrix.columns:
        return np.nan
    
    # Öğenin benzerlik skorlarını al
    item_similarities = item_similarity_df[item_id]
    
    # Öğenin benzerlik skorlarına göre sıralama
    similar_items = item_similarities.sort_values(ascending=False).index[1:k+1]
    
    # Benzer öğelerin kullanıcı değerlendirmelerini al
    similar_items_ratings = item_user_matrix.loc[similar_items, user_id]
    
    # Tahmini hesapla
    prediction = np.dot(item_similarities[similar_items], similar_items_ratings) / np.sum(item_similarities[similar_items])
    return prediction

def calculate_rmse_item_based(k):
    test_data['prediction'] = test_data.apply(lambda x: predict_item_based(x['user_id'], x['item_id'], k), axis=1)
    valid_predictions = test_data['prediction'].notna()
    filtered_test_data = test_data[valid_predictions]
    rmse = np.sqrt(mean_squared_error(filtered_test_data['rating'], filtered_test_data['prediction']))
    return rmse

# Farklı K değerleri için RMSE hesaplama
k_values = [5, 10, 20, 30, 40, 50, 100, 150]
results = []

for k in k_values:
    user_rmse = calculate_rmse_user_based(k)
    item_rmse = calculate_rmse_item_based(k)
    results.append({
        "Yöntem": "User-KNN",
        "Hiperparametreler": f"K = {k}, cosine",
        "Test RMSE": user_rmse
    })
    results.append({
        "Yöntem": "Item-KNN",
        "Hiperparametreler": f"K = {k}, cosine",
        "Test RMSE": item_rmse
    })

# Sonuçları tablo şeklinde sunma
results_df = pd.DataFrame(results)
print(results_df)