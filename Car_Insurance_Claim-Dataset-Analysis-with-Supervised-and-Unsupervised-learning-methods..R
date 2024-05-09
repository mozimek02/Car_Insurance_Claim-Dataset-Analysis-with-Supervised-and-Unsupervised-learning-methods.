
# Packages -----------------------------------------------------------------
library(Matrix)
library(xgboost)
library(e1071)
library(neuralnet)
library(gbm)
library(adabag)
library(randomForest)
library(rpart)
library(tidyverse)
library(dplyr)
library(caTools)
library(rattle)
library(caret)
library(ggplot2)
library(pROC)
# Data correction ----------------------------------------------------------

dane <- read.csv("Car_Insurance_Claim.csv")

ktore_braki <- function(x){
  for(i in colnames(x)){
    if((any(is.na(x[[i]]) == TRUE)) == TRUE){
      print(i)
    }
  }
}
ktore_braki(dane)
#"CREDIT_SCORE","ANNUAL_MILEAGE"

podstaw <- function(x){
  for(i in 1:10000){
    if(is.na(x$CREDIT_SCORE[i]) == TRUE){
      x$CREDIT_SCORE[i] = mean(na.omit(x$CREDIT_SCORE))
    }
  }
  return(x)
}
dane_pop <- podstaw(dane)

podstaw2 <- function(x){
  for(i in 1:10000){
    if(is.na(x$ANNUAL_MILEAGE[i]) == TRUE){
      x$ANNUAL_MILEAGE[i] = mean(na.omit(x$ANNUAL_MILEAGE))
    }
  }
  return(x)
}
dane_dobre <- podstaw2(dane_pop)
cbind("CREDIT_SCORE" = any(dane_dobre$CREDIT_SCORE == TRUE),"ANNUAL_MILEAGE" = any(dane_dobre$ANNUAL_MILEAGE == TRUE))

# Classification using one model. ----------------------------------
#At the beginning of the project, I will check how a single classifier works and then compare its quality with the team classifiers.

dane_dobre$OUTCOME <- factor(dane_dobre$OUTCOME)
levels(dane_dobre$OUTCOME) <- c("Dostanie","Nie dostanie")
set.seed(456)
podzial = sample.split(dane_dobre$OUTCOME, SplitRatio = 0.7)
zbior_treningowy <- subset(dane_dobre, podzial == TRUE)
zbior_testowy <- subset(dane_dobre, podzial == FALSE)
model <- rpart(formula = OUTCOME~ .,data = zbior_treningowy)
fancyRpartPlot(model)

predykcja <- predict(model, newdata = zbior_testowy, type = "class")
confusionMatrix(predykcja, zbior_testowy$OUTCOME, mode = "everything", positive="Dostanie")
#The F-measure is 0.8784 which means that the model is quite good.


# Team classification - Random Forest -------------------------------------
set.seed(456)
las_losowy <- randomForest(OUTCOME~.,data=zbior_treningowy,ntree=1000)
predykcja_las <- predict(las_losowy,newdata = zbior_testowy, type="class")
confusionMatrix(predykcja_las, zbior_testowy$OUTCOME, mode = "everything", positive="Dostanie")
#The F-measure value in the case of random forest was 0.8914.
#So we can conclude that the random forest based classifier is better.

#OOB mean square error plot in random forest.
las_losowy$err.rate
plot(las_losowy)
las_losowy.legenda <- colnames(las_losowy$err.rate)
legend(x=300,y=0.5,legend = las_losowy.legenda,lty=c(1,2,3),
       col=c(1,2,3))
min.err <- min(las_losowy$err.rate[,"OOB"])
min.err
min.err.idx <- which(las_losowy$err.rate[,"OOB"]== min.err)
min.err.idx
las_losowy$err.rate[min.err.idx,]
#Looking at the error graph, it can be concluded that the OOB error is quite high, but then it drops quickly and is stable.

# Reinforcement learning -------------------------------------------------

#Will use the Adaboost model.
wzmocnienie <- boosting(OUTCOME~., zbior_treningowy,mfinal = 50)
wzmocnienie$importance
predykcja_ada <- predict(wzmocnienie,newdata = zbior_testowy, type="class")
table(predykcja_ada$class,zbior_testowy$OUTCOME)
confusionMatrix(as.factor(predykcja_ada$class), zbior_testowy$OUTCOME, mode = "everything", positive="Dostanie")
#The F-measure value was 0.8860. It is smaller than for a team learner based on bagging without reinforcement.

#Gradient methods.
#Re-preparing data and converting it to numerical variables.
dane_dobre_dv <- dummyVars("~ .",dane_dobre[-19], fullRank = T)
dane_dobre_dv_df <- as.data.frame(predict(dane_dobre_dv,newdata = dane_dobre[-19]))
dane_dobre_dv_df <- cbind(dane_dobre_dv_df,dane_dobre[19])
levels(dane_dobre_dv_df$OUTCOME) <- c(1,0)
str(dane_dobre_dv_df)
set.seed(456)
View(dane_dobre_dv_df)
podzial2 <- sample.split(dane_dobre_dv_df$OUTCOME,SplitRatio = 0.7)
dane_dobre_dv_df_train <- subset(dane_dobre_dv_df,podzial2 == TRUE)
dane_dobre_dv_df_test <- subset(dane_dobre_dv_df,podzial2 == FALSE)


#GBM
set.seed(456)
levels(dane_dobre_dv_df_test$OUTCOME)<-c("Dostanie","Nie dostanie")
gradient <- gbm(OUTCOME~.,distribution = "gaussian",data = dane_dobre_dv_df_train,
                n.trees = 1000,shrinkage = 0.05, cv.folds = 2)
liczba_drzew <- gbm.perf(gradient)
gradient_predykcje <- predict(gradient,dane_dobre_dv_df_test,n.trees=liczba_drzew)
gbm.roc = roc(dane_dobre_dv_df_test$OUTCOME, gradient_predykcje)
x=plot(gbm.roc)
x
coords(gbm.roc, "best")
gradient_klasy <- ifelse(gradient_predykcje < 1.400101, "Dostanie", "Nie dostanie")
confusionMatrix(as.factor(gradient_klasy), dane_dobre_dv_df_test$OUTCOME, mode = "everything", positive="Dostanie")
table(gradient_klasy,dane_dobre_dv_df_test$OUTCOME)
#F-measure is: 0.8771.

#XGBM <- improved version of GBM
levels(dane_dobre_dv_df_test$OUTCOME)<-c(1,0)
dane.macierz <- as.matrix(dane_dobre_dv_df_train[,-26])
trening.macierz <- as(dane.macierz,"dgCMatrix")
dane2.macierz<- as.matrix(dane_dobre_dv_df_test[,-26])
test.macierz <- as(dane2.macierz,"dgCMatrix")
dane_dobre_dv_df_train$OUTCOME <- as.factor(dane_dobre_dv_df_train$OUTCOME)
xgb_gradient <- xgboost::xgboost(data=trening.macierz,label = dane_dobre_dv_df_train$OUTCOME,
                        nrounds = 1000,
                        eval_metric = "logloss")
xgb.predykcja <- predict(xgb_gradient,test.macierz)
gbm.roc = roc(dane_dobre_dv_df_test$OUTCOME, xgb.predykcja)
x=plot(gbm.roc)
x
coords(gbm.roc, "best")
xgb_klasy <- ifelse(xgb.predykcja < 1.309854, "Dostanie", "Nie dostanie")
levels(dane_dobre_dv_df_test$OUTCOME)<-c("Dostanie","Nie dostanie")
confusionMatrix(as.factor(xgb_klasy), dane_dobre_dv_df_test$OUTCOME, mode = "everything", positive="Dostanie")
table(xgb_klasy,dane_dobre_dv_df_test$OUTCOME)
#F-measure is: 0.8436.
#The method uses logistic regression for classification, set by default in the xgboost function.

# Neuron network ----------------------------------------------------------
#I will use a neural network to classify the previously used data. This time I will choose only 4 variables. Neural network students require the most
#operations, which would extend the working time compared to other classifiers.
set.seed(456)
levels(dane_dobre_dv_df_train$OUTCOME) <- c("Dostanie","Nie dostanie")
model_sieci <- neuralnet(OUTCOME~VEHICLE_OWNERSHIP+PAST_ACCIDENTS+SPEEDING_VIOLATIONS+DUIS,data=dane_dobre_dv_df_train,hidden=2,lifesign="full", lifesign.step=5000)
plot(model_sieci)
#Model evaluation
siec_predykcja <- neuralnet::compute(model_sieci,dane_dobre_dv_df_test[-26])
str(siec_predykcja)
najlepsza.kolumna <- apply(siec_predykcja$net.result,1,which.max)
siec.predykcje = c("Dostanie","Nie dostanie")[najlepsza.kolumna]
#Confusion matrix
levels(dane_dobre_dv_df_test)<-c("Dostanie","Nie dostanie")
table(dane_dobre_dv_df_test$OUTCOME,siec.predykcje)
nrow(dane_dobre_dv_df_test)
confusionMatrix(as.factor(siec.predykcje), dane_dobre_dv_df_test$OUTCOME, mode = "everything", positive="Dostanie")
#F-measure is: 0.8603.


#I will perform another classification using the SVM algorithm, this time on the voice.csv data set.
set.seed(456)
dane2 <- read.csv("voice.csv")
dane2$label <- as.factor(dane2$label)
podzial_voice <- sample(c("Train","Test"),nrow(dane2),replace = TRUE, prob = c(0.7,0.3))
voice_trening <- subset(dane2,podzial_voice == "Train")
voice_test <- subset(dane2,podzial_voice == "Test")

model_svm <- svm(label~ .,data = voice_trening, kernel = "sigmoid", type = "C")
svm.predykcje <- predict(model_svm, voice_test)
svm_tabelka <- table(svm.predykcje,voice_test$label)
svm_tabelka
confusionMatrix(svm.predykcje, voice_test$label, mode = "everything", positive="male")
#F-measure is: 0.8004.


# Podsumowanie ------------------------------------------------------------

#We can conclude that both the single classifier and the composite classifier achieved similar error scores for the Car_Insurance_Claim dataset.
#On average, the classifiers' score was...
wynik <- mean(0.8784,0.8914,0.8860,0.87717,0.8436)
wynik
#Average F-measure score 0.8784
#The neural network on a smaller number of variables from the Car_Insurance_Claim set and SVM on voice.csv data also achieved F-measure values
#above 0.8.
#We can therefore conclude that the data used above are suitable for classification using various methods.
#Let's see the time efficiency of the neural network and SVM.
siec_test <- function(x){
  siec_test_wynik <- neuralnet(OUTCOME~VEHICLE_OWNERSHIP+PAST_ACCIDENTS+SPEEDING_VIOLATIONS+DUIS,data=dane_dobre_dv_df_train,hidden=2,lifesign="full", lifesign.step=5000)
  return(siec_test_wynik)
}

svm_test <- function(x){
  svm_test_wynik <- svm(label~ .,data = voice_trening, kernel = "sigmoid", type = "C")
  return(svm_test_wynik)
}

start_time <- Sys.time()
svm_test
end_time <- Sys.time()
end_time - start_time
#Time difference of 0.001791954 secs

start_time <- Sys.time()
siec_test
end_time <- Sys.time()
end_time - start_time
#Time difference of 0.001698017 secs

