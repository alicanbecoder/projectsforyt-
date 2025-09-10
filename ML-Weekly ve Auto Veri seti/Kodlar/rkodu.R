library(ISLR)
data(Weekly)
summary(Weekly)

# Histogram: Volume
hist(Weekly$Volume, main="Volume Histogram", xlab="Volume", col="skyblue", border="white")

# Boxplot: Lag deðiþkenleri
boxplot(Weekly[,2:6], main="Lag Deðiþkenleri Boxplot")

plot(Weekly$Volu?e, type="l", main="Volume Zaman Serisi", ylab="Volume", xlab="Hafta")

pairs(Weekly[,2:6], col=ifelse(Weekly$Direction=="Up", "green", "red"), main="Lag Deðiþkenleri ve Yön")

# Lojistik regresyon modeli (binomial aile)
model <- glm(Direction ~ Lag1 + Lag2?+ Lag3 + Lag4 + Lag5 + Volume, 
             data = Weekly, 
             family = binomial)

cor(Weekly[-9])

# Model sonuçlarý
summary(model)

# Modelden olasýlýk tahmini yapýyoruz
prob <- predict(model, type = "response")

# Olasýlýða göre tahmini sýnýf?arý belirliyoruz (eþik: 0.5)
predicted_direction <- ifelse(prob > 0.5, "Up", "Down")
predicted_direction <- factor(predicted_direction, levels = c("Down", "Up"))

# Gerçek sýnýflar
actual_direction <- Weekly$Direction

# Conf matrisi oluþturma
table(Predic?ed = predicted_direction, Actual = actual_direction)

conf_matrix <- table(Predicted = predicted_direction, Actual = actual_direction)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
accuracy

# Eðitim: 1990-2008
train <- subset(Weekly, Year < 2009)
?# Test: 2009-2010
test <- subset(Weekly, Year >= 2009)

model_lag2 <- glm(Direction ~ Lag2, data = train, family = binomial)
summary(model_lag2)

# Test için olasýlýk tahmini
prob_test <- predict(model_lag2, newdata = test, type = "response")

# Tahmini sý?ýflar (eþik 0.5)
pred_test <- ifelse(prob_test > 0.5, "Up", "Down")
pred_test <- factor(pred_test, levels = c("Down", "Up"))

# Gerçek test sýnýflarý
actual_test <- test$Direction

# Karýþýklýk matrisi
conf_matrix_test <- table(Predicted = pred_test, Actua? = actual_test)
print(conf_matrix_test)

# Doðru sýnýflandýrma oraný
accuracy_test <- sum(diag(conf_matrix_test)) / sum(conf_matrix_test)
accuracy_test

