# Required
library(doParallel)
library(plotly)
library(corrplot)
library(xgboost)
library(caret)
# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6
# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers()

iPhone <- read.csv("~/Desktop/Data Science/ubiqum/projects/DS4/Task 2/iphone_smallmatrix_labeled_8d.csv")
samsung <- read.csv("~/Desktop/Data Science/ubiqum/projects/DS4/Task 2/galaxy_smallmatrix_labeled_8d.csv")
large <- read.csv("~/Desktop/Data Science/ubiqum/projects/DS4/Task 2/LargeMatrix.csv")

large <- large[, -which(names(large) %in% c("X", "id"))]

# Examine Feature Variance
nzv <- nearZeroVar(iPhone, saveMetrics = FALSE) 
iphoneNZV <- iPhone[,-nzv]
str(iphoneNZV)
options(max.print=1000000)

nzv_sams <- nearZeroVar(samsung, saveMetrics = FALSE) 
samsungNZV <- samsung[,-nzv_sams]
names(samsungNZV)

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC <-iPhone
iphoneRC$iphonesentiment <- dplyr::recode(iPhone$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

samsungRC <-samsung
samsungRC$galaxysentiment <- dplyr::recode(samsungRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

# make sentiments a factor
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)
samsungRC$galaxysentiment <- as.factor(samsungRC$galaxysentiment)

#Spliiting the data in to train and test set
#Iphone
n = nrow(iphoneRC)
train.index = sample(n,floor(0.7*n))
train.data_i = iphoneRC[train.index,]
test.data_i = iphoneRC[-train.index,]
#Smasung
train.data_s = samsungRC[train.index,]
test.data_s = samsungRC[-train.index,]

#PCA
preprocessParams_i <- preProcess(train.data_i[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams_i)

preprocessParams_s <- preProcess(train.data_s[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams_s)

# use predict to apply pca parameters, create training, exclude dependant
train.pca_i <- predict(preprocessParams_i, train.data_i[,-59])
train.pca_s <- predict(preprocessParams_s, train.data_s[,-59])

# add the dependent to training
train.pca_i$iphonesentiment <- train.data_i$iphonesentiment
train.pca_s$galaxysentiment <- train.data_s$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca_i <- predict(preprocessParams_i, test.data_i[,-59])
test.pca_s <- predict(preprocessParams_s, test.data_s[,-59])

#Large Matric using pca
large.pca_i <-predict(preprocessParams_i, large)

# add the dependent to training
test.pca_i$iphonesentiment <- test.data_i$iphonesentiment
test.pca_s$galaxysentiment <- test.data_s$galaxysentiment

# inspect results
str(train.pca_i)
str(train.pca_s)

#Preparing for XGBoost

iPhoneXG <- rbind(train.pca_i, test.pca_i)
samsungXG <- rbind(train.pca_s, test.pca_s)
iphonesentiment <- iPhoneXG$iphonesentiment
galaxysentiment <- samsungXG$galaxysentiment

label_i = as.integer(iPhoneXG$iphonesentiment) -1
iPhoneXG$iphonesentiment <- NULL

label_s = as.integer(samsungXG$galaxysentiment) -1
samsungXG$galaxysentiment <- NULL

n = nrow(iPhoneXG)
#Iphone
train.index_i = sample(n,floor(0.7*n))
train.data_i = as.matrix(iPhoneXG[train.index_i,])
train.label_i = label_i[train.index_i]
test.data_i = as.matrix(iPhoneXG[-train.index_i,])
test.label_i = label_i[-train.index_i]

#Samsung
train.index_s = sample(n,floor(0.7*n))
train.data_s = as.matrix(samsungXG[train.index_s,])
train.label_s = label_s[train.index_s]
test.data_s = as.matrix(samsungXG[-train.index_s,])
test.label_s = label_s[-train.index_s]

#large matrix iPhone
large_i = as.matrix(large.pca_i)

# Transform the two data sets into xgb.Matrix
#iphone
xgb.train_i = xgb.DMatrix(data=train.data_i,label=train.label_i)
xgb.test_i = xgb.DMatrix(data=test.data_i,label=test.label_i)
#Samsung
xgb.train_s = xgb.DMatrix(data=train.data_s,label=train.label_s)
xgb.test_s = xgb.DMatrix(data=test.data_s,label=test.label_s)

# Define the parameters for multinomial classification
#iPhone
num_class_i = length(levels(iphonesentiment))
params_i = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.7,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class_i
)

#samsung
num_class_s = length(levels(galaxysentiment))
params_s = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.7,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class_s
)

# Train the XGBoost classifer
#iPhone
xgb.fit_i=xgb.train(
  params=params_i,
  data=xgb.train_i,
  nrounds=3000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train_i,val2=xgb.test_i),
  verbose=0
)

#Samsung

xgb.fit_s=xgb.train(
  params=params_s,
  data=xgb.train_s,
  nrounds=3000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train_s,val2=xgb.test_s),
  verbose=0
)

# Review the final model and results 
xgb.fit_i
xgb.fit_s

# Predict outcomes with the test data

#iPhone
xgb.pred_i = predict(xgb.fit_i,test.data_i,reshape=T)
xgb.pred_i = as.data.frame(xgb.pred_i)
colnames(xgb.pred_i) = levels(iphonesentiment)
#Samsung
xgb.pred_s = predict(xgb.fit_s,test.data_s,reshape=T)
xgb.pred_s = as.data.frame(xgb.pred_s)
colnames(xgb.pred_s) = levels(galaxysentiment)

#Large
xgb.large_i = predict(xgb.fit_i, large_i, reshape = T)
xgb.large_i = as.data.frame(xgb.large_i)
colnames(xgb.large_i) = levels(iphonesentiment)

# Use the predicted label with the highest probability
#iPhone
xgb.pred_i$prediction = apply(xgb.pred_i,1,function(x) colnames(xgb.pred_i)[which.max(x)])
xgb.pred_i$label = levels(iphonesentiment)[test.label_i+1]
#Samsung
xgb.pred_s$prediction = apply(xgb.pred_s,1,function(x) colnames(xgb.pred_s)[which.max(x)])
xgb.pred_s$label = levels(galaxysentiment)[test.label_s+1]

#Large
xgb.large_i$prediction = apply(xgb.large_i,1,function(x) colnames(xgb.large_i)[which.max(x)])

# Calculate the final accuracy iPhone
Accuracy_i = sum(xgb.pred_i$prediction==xgb.pred_i$label)/nrow(xgb.pred_i)
print(paste("Final Accuracy iPhone =",sprintf("%1.2f%%", 100*Accuracy_i))) # Accuracy 84,84%
confusion_i <- table(true = xgb.pred_i$label, pred = xgb.pred_i$prediction)

# Calculate the final accuracy Samsung
Accuracy_s = sum(xgb.pred_s$prediction==xgb.pred_s$label)/nrow(xgb.pred_s)
print(paste("Final Accuracy Samsung =",sprintf("%1.2f%%", 100*Accuracy_s))) # Accuracy 86,54%
confusion_s <- table(true = xgb.pred_s$label, pred = xgb.pred_s$prediction)

#Confusion Matrix for xgboost 
#iPhone
xgb.pred_i$prediction <- as.factor(xgb.pred_i$prediction)
xgb.pred_i$label <- as.factor(xgb.pred_i$label)
conf_mat_i_xgb<-confusionMatrix(xgb.pred_i$prediction, xgb.pred_i$label) 
#Samsung

xgb.pred_s$prediction <- as.factor(xgb.pred_s$prediction)
xgb.pred_s$label <- as.factor(xgb.pred_s$label)
conf_mat_s_xgb<-confusionMatrix(xgb.pred_s$prediction, xgb.pred_s$label) 

# OTHER MODELS
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = TRUE)

#SVM Accuracy: 0.767, Kappa: 0.372, time: 488 sec
system.time(iPhone_SVM<-caret::train(iphonesentiment~., data= train.pca_i, method="svmLinear", 
                                         trControl=fitControl, allowParallel=TRUE))

Predict_iPhone_svm<-predict(iPhone_SVM, test.pca_i)
conf_mat_i_SVM<-confusionMatrix(Predict_iPhone_svm, test.pca_i$iphonesentiment) 
conf_mat_i_SVM

#Random Forest Accuracy: 0.846, Kappa: 0.618, time:850 sec
system.time(iPhone_rf<-caret::train(iphonesentiment~., data= train.pca_i, method="rf", 
                                     trControl=fitControl, allowParallel=TRUE))

Predict_iPhone_rf<-predict(iPhone_rf, test.pca_i)
conf_mat_i_rf<-confusionMatrix(Predict_iPhone_rf, test.pca_i$iphonesentiment) 
conf_mat_i_rf

#С5 Accuracy: 0.838, Kappa: 0.597, time: 183 sec
system.time(iPhone_c5<-caret::train(iphonesentiment~., data= train.pca_i, method="C5.0", 
                                    trControl=fitControl, allowParallel=TRUE))

Predict_iPhone_c5<-predict(iPhone_c5, test.pca_i)
conf_mat_i_c5<-confusionMatrix(Predict_iPhone_c5, test.pca_i$iphonesentiment) 
conf_mat_i_c5

#KNN Accuracy: 0.8386, Kappa: 0.602, time: 35.96
library(kknn)
system.time(iPhone_KNN <- train.kknn(iphonesentiment~., data = train.pca_i, k = 25, kernel = c("rectangular", "triangular", "epanechnikov",
                                                                                               "gaussian", "rank", "optimal")))
Predict_iPhone_knn<-predict(iPhone_KNN, test.pca_i)
conf_mat_i_knn<-confusionMatrix(Predict_iPhone_knn, test.pca_i$iphonesentiment) 
conf_mat_i_knn

save.image(file='iPhone_Samsung_sentiment')
#load('iPhone_Samsung_sentiment')



