# Required
library(doParallel)
library(plotly)
library(corrplot)
library(xgboost)
# Find how many cores are on your machine
detectCores() # Result = Typically 4 to 6
# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

Iphone <- read.csv("~/Desktop/Data Science/ubiqum/projects/DS4/Task 2/iphone_smallmatrix_labeled_8d.csv")

summary(Iphone)
names(Iphone)
str(Iphone)
anyNA(Iphone)

#Visualization
plot_ly(Iphone, x= ~Iphone$iphonesentiment, type='histogram')
plot_ly(Iphone, x= ~Iphone$ios, type='histogram')
plot_ly(Iphone, x= ~Iphone$iphonecampos, type='histogram')
plot_ly(Iphone, x= ~Iphone$iphonecamneg, type='histogram')
plot_ly(Iphone, x= ~Iphone$iphonedispos, type='histogram')

# Corellation Matrix
IohoneCOR <- cor(x = Iphone,y = NULL, use = "everything",
         method = c("pearson", "kendall", "spearman"))

corrplot(IohoneCOR, method = "color")

# Examine Feature Variance
nzv <- nearZeroVar(Iphone, saveMetrics = FALSE) 
iphoneNZV <- Iphone[,-nzv]
str(iphoneNZV)
options(max.print=1000000)

# Let's sample the data before using RFE
set.seed(123)
iphoneSample <- Iphone[sample(1:nrow(Iphone), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"))

save.image(file='smartphone sentiment.RData')
load('smartphone sentiment.RData')

#Creating new data set with best features
newIphone <- subset(Iphone, select=c("iphone", "googleandroid", "iphonedispos", "iphonedisneg", "samsunggalaxy", "htcphone", "iphonedisunc",
       "iphoneperpos", "ios", "iphoneperneg", "sonyxperia", "iphoneperunc", "iphonecampos", "iphonecamneg",
       "iphonecamunc","htcdisunc", "htccampos", "htcperpos","htccamneg","iphonesentiment"))

newIphone$iphonesentiment <- as.factor(newIphone$iphonesentiment)

iphonesentiment <- newIphone$iphonesentiment
label = as.integer(newIphone$iphonesentiment) -1
newIphone$iphonesentiment <- NULL

#Spliiting the data in to train and test set

n = nrow(newIphone)
train.index = sample(n,floor(0.7*n))
train.data = as.matrix(newIphone[train.index,])
train.label = label[train.index]
test.data = as.matrix(newIphone[-train.index,])
test.label = label[-train.index]

# Transform the two data sets into xgb.Matrix
xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)

# Define the parameters for multinomial classification
num_class = length(levels(iphonesentiment))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.7,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=5000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=0
)

# Review the final model and results 
xgb.fit

# Predict outcomes with the test data
xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(iphonesentiment)

# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(iphonesentiment)[test.label+1]

# Calculate the final accuracy
result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result))) # Accuracy 78,37%

# More Feature Engeneering

#Engineering the Dependant variable
#create a new dataset that will be used for recoding sentiment
iphoneRC <-Iphone
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphoneRC$iphonesentiment <- recode(Iphone$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

n = nrow(iphoneRC)
train.index = sample(n,floor(0.7*n))
train.data = iphoneRC[train.index,]
test.data = iphoneRC[-train.index,]

preprocessParams <- preProcess(train.data[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)


# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, train.data[,-59])

# add the dependent to training
train.pca$iphonesentiment <- train.data$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, test.data[,-59])

# add the dependent to training
test.pca$iphonesentiment <- test.data$iphonesentiment

confusionMatrix(as.factor(test.data$iphonesentiment), test.pca$iphonesentiment)

# inspect results
str(train.pca)
str(test.pca)



# make iphonesentiment a factor
train.pca$iphonesentiment <- as.factor(train.pca$iphonesentiment)
test.pca$iphonesentiment <- as.factor(test.pca$iphonesentiment)


iphonesentiment <- iphoneRC$iphonesentiment
label = as.integer(iphoneRC$iphonesentiment) -1
iphoneRC$iphonesentiment <- NULL

#Spliiting the data in to train and test set

n = nrow(iphoneRC)
train.index = sample(n,floor(0.7*n))
train.data = as.matrix(newIphone[train.index,])
train.label = label[train.index]
test.data = as.matrix(newIphone[-train.index,])
test.label = label[-train.index]

# Transform the two data sets into xgb.Matrix
xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)

# Define the parameters for multinomial classification
num_class = length(levels(iphonesentiment))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.7,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=3000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=0
)

# Review the final model and results 
xgb.fit

# Predict outcomes with the test data
xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(iphonesentiment)

# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(iphonesentiment)[test.label+1]

# Calculate the final accuracy
result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result))) # Accuracy 85,35% with all 84,25%


# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)
