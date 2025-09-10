# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:20:28 2024

@author: Alican
"""


from dataset import create_dataset, plot_dataset
from neural_network import evaluate_performance_ann, loss_models,plot_loss,plot_decision_boundary_ann
from svm import train_svm, plot_decision_boundary,train_with_params
from svm import evaluate_performance,plot_train_wp,save_results_to_txt

X,y,X_train, X_val, X_test, y_train, y_val, y_test = create_dataset()
datasets = (X_test, y_test, X_val, y_val, X_train, y_train)
plot_dataset(X,y)


#Yapay nörön ağları


# Model oluşturma (örnek: 1 gizli katman) 
input_dim = X_train.shape[1]  # X_train'deki özellik sayısını alıyoruz (2 ise, input_dim=2)
hidden_layers = [20]
kernel = 'sigmoid'
model_name = f"model_{kernel}_input{input_dim}_hidden{hidden_layers[0]}"
epochs=500


history_sgd,model_sgd,history_bgd,model_bgd,history_mbgd,model_mbgd=loss_models(input_dim, hidden_layers, X_train, y_train, X_val, y_val, epochs)

plot_loss(history_sgd,history_bgd,history_mbgd)



# Performansı değerlendirme ve dosyaya kaydetme
evaluate_performance_ann(model_sgd, X_train, X_val, X_test, y_train, y_val, y_test,model_name)
evaluate_performance_ann(model_bgd, X_train, X_val, X_test, y_train, y_val, y_test,model_name)
evaluate_performance_ann(model_mbgd, X_train, X_val, X_test, y_train, y_val, y_test,model_name)
plot_decision_boundary_ann(model_sgd, X, y)
plot_decision_boundary_ann(model_bgd, X, y)
plot_decision_boundary_ann(model_mbgd, X, y)


#2 katmanlı ağ

input_dim2 = X_train.shape[1]  # X_train'deki özellik sayısını alıyoruz (2 ise, input_dim=2)
hidden_layers2 = [10,10]  # İki katman ve 10 nöron
kernel = 'sigmoid'
model_name2 = f"model_{kernel}_input{input_dim2}_hidden{hidden_layers2[0]}"
epochs2=500

history_sgd2,model_sgd2,history_bgd2,model_bgd2,history_mbgd2,model_mbgd2=loss_models(input_dim2, hidden_layers2, X_train, y_train, X_val, y_val, epochs2)
plot_loss(history_sgd2,history_bgd2,history_mbgd2) 


evaluate_performance_ann(model_sgd2, X_train, X_val, X_test, y_train, y_val, y_test,model_name2)
evaluate_performance_ann(model_bgd2, X_train, X_val, X_test, y_train, y_val, y_test,model_name2)
evaluate_performance_ann(model_mbgd2, X_train, X_val, X_test, y_train, y_val, y_test,model_name2)
plot_decision_boundary_ann(model_sgd2, X, y)
plot_decision_boundary_ann(model_bgd2, X, y)
plot_decision_boundary_ann(model_mbgd2, X, y)



#3 katmanlı norön ağı

input_dim3 = X_train.shape[1]  # X_train'deki özellik sayısını alıyoruz (2 ise, input_dim=2)
hidden_layers3 = [10,10,10]  # Üç katman ve 10 nöron
kernel = 'sigmoid'
model_name3 = f"model_{kernel}_input{input_dim3}_hidden{hidden_layers3[0]}"
epochs3=1000

history_sgd3,model_sgd3,history_bgd3,model_bgd3,history_mbgd3,model_mbgd3=loss_models(input_dim3, hidden_layers3, X_train, y_train, X_val, y_val, epochs3)
plot_loss(history_sgd3,history_bgd3,history_mbgd3) 


evaluate_performance_ann(model_sgd3, X_train, X_val, X_test, y_train, y_val, y_test,model_name3)
evaluate_performance_ann(model_bgd3, X_train, X_val, X_test, y_train, y_val, y_test,model_name3)
evaluate_performance_ann(model_mbgd3, X_train, X_val, X_test, y_train, y_val, y_test,model_name3)
plot_decision_boundary_ann(model_sgd3, X, y)
plot_decision_boundary_ann(model_bgd3, X, y)
plot_decision_boundary_ann(model_mbgd3, X, y)


# SVM model eğitimi linear kernel için

kernel = 'linear'
model_name = f"svm_model_{kernel}"
train_accuracies, val_accuracies, train_losses, val_losses, C_values = train_with_params(X_train, y_train, X_val, y_val,kernel)
plot_train_wp(train_accuracies, val_accuracies, train_losses, val_losses, C_values)
best_C=1
# SVM modelinin en iyi paremetlere göre sonuçları
svm_model_lr = train_svm(X_train, y_train, kernel , C=best_C)
results=evaluate_performance(svm_model_lr, X_train, X_val, X_test, y_train, y_val, y_test, best_C)
save_results_to_txt(results, model_name)
#Karar çzigisi
plot_decision_boundary(svm_model_lr, X, y)

# SVM model eğitimi linear polinomsal için
kernel = 'poly'
model_name = f"svm_model_{kernel}"
train_accuracies, val_accuracies, train_losses, val_losses, C_values = train_with_params(X_train, y_train, X_val, y_val,kernel)
plot_train_wp(train_accuracies, val_accuracies, train_losses, val_losses, C_values)
best_C=110
# SVM modelinin en iyi paremetlere göre sonuçları
svm_model_lr = train_svm(X_train, y_train, kernel , C=best_C)
results=evaluate_performance(svm_model_lr, X_train, X_val, X_test, y_train, y_val, y_test, best_C)
save_results_to_txt(results, model_name)
#Karar çzigisi
plot_decision_boundary(svm_model_lr, X, y)

# SVM model eğitimi linear Gaussian RBF için
kernel = 'rbf'
model_name = f"svm_model_{kernel}"
train_accuracies, val_accuracies, train_losses, val_losses, C_values = train_with_params(X_train, y_train, X_val, y_val,kernel)
plot_train_wp(train_accuracies, val_accuracies, train_losses, val_losses, C_values)
best_C=5
# SVM modelinin en iyi paremetlere göre sonuçları
svm_model_lr = train_svm(X_train, y_train, kernel , C=best_C)
results=evaluate_performance(svm_model_lr, X_train, X_val, X_test, y_train, y_val, y_test, best_C)
save_results_to_txt(results, model_name)
#Karar çzigisi
plot_decision_boundary(svm_model_lr, X, y)
