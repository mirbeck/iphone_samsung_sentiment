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

set.seed(123)

# Examine Feature Variance
nzv <- nearZeroVar(iPhone, saveMetrics = FALSE) 
iphoneNZV <- iPhone[,-nzv]
str(iphoneNZV)
options(max.print=1000000)

# recode sentiment to combine factor levels 1 negative 2 neutral 3 positive
iphoneRC <-iphoneNZV
iphoneRC$iphonesentiment <- dplyr::recode(iPhone$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3) 

#Spliiting the data in to train and test set
#Iphone
n = nrow(iphoneRC)
train.index = sample(n,floor(0.7*n))
train.data_i = iphoneRC[train.index,]
test.data_i = iphoneRC[-train.index,]

### Balancing train set

df1 <- filter(train.data_i, iphonesentiment %in% c(1, 3))
df2 <- filter(train.data_i, iphonesentiment %in% c(2, 3))
df3 <- filter(train.data_i, iphonesentiment %in% c(3))

library(ROSE)

df1_over <- ovun.sample(iphonesentiment~., data = df1, method = "over", N = 12624)$data
table(df1_over$iphonesentiment)

df1 <- filter(df1_over, iphonesentiment %in% c(1))

df2_over <- ovun.sample(iphonesentiment~., data = df2, method = "over", N = 12624)$data
table(df2_over$iphonesentiment)

df2 <- filter(df2_over, iphonesentiment %in% c(2))

iphone_bal <- rbind(df1, df2, df3)

prop.table(table(iphone_bal$iphonesentiment))
barplot(prop.table(table(iphone_bal$iphonesentiment)),
        col = rainbow(4),
        ylim = c(0,0.5),
        main = "Class Distribution after ROSE")


train.data_i <- iphone_bal


#PCA
preprocessParams_i <- preProcess(train.data_i[,-12], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams_i)

# use predict to apply pca parameters, create training, exclude dependant
train.pca_i <- predict(preprocessParams_i, train.data_i[,-12])

# add the dependent to training
train.pca_i$iphonesentiment <- train.data_i$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
test.pca_i <- predict(preprocessParams_i, test.data_i[,-12])

# add the dependent to training
test.pca_i$iphonesentiment <- test.data_i$iphonesentiment

# inspect results
str(train.pca_i)
str(test.pca_i)

#Delete real values
train_iphonesentiment<- train.pca_i$iphonesentiment
test_iphonesentiment <- test.pca_i$iphonesentiment
train.pca_i$iphonesentiment <- NULL
test.pca_i$iphonesentiment <- NULL

train.data_i = as.matrix(train.pca_i)
train.label_i = as.integer(train_iphonesentiment)-1
test.data_i = as.matrix(test.pca_i)
test.label_i = as.integer(test_iphonesentiment)-1

# Transform the two data sets into xgb.Matrix
#iphone
xgb.train_i = xgb.DMatrix(data=train.data_i,label=train.label_i)
xgb.test_i = xgb.DMatrix(data=test.data_i,label=test.label_i)

# Define the parameters for multinomial classification
#iPhone
iphonesentiment <-as.factor(iphoneRC$iphonesentiment)

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

# Train the XGBoost classifer
#iPhone
system.time(xgb.fit_i=xgb.train(
  params=params_i,
  data=xgb.train_i,
  nrounds=1000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train_i,val2=xgb.test_i),
  verbose=0
))

# Review the final model and results 
xgb.fit_i
# Predict outcomes with the test data

#iPhone
xgb.pred_i = predict(xgb.fit_i,test.data_i,reshape=T)
xgb.pred_i = as.data.frame(xgb.pred_i)
colnames(xgb.pred_i) = levels(iphonesentiment)

# Use the predicted label with the highest probability

#iPhone
xgb.pred_i$prediction = apply(xgb.pred_i,1,function(x) colnames(xgb.pred_i)[which.max(x)])
xgb.pred_i$label = levels(iphonesentiment)[test.label_i+1]

# Calculate the final accuracy iPhone
Accuracy_i = sum(xgb.pred_i$prediction==xgb.pred_i$label)/nrow(xgb.pred_i)
print(paste("Final Accuracy iPhone =",sprintf("%1.2f%%", 100*Accuracy_i))) # Accuracy  76.23%, kappa: 0.501

#Confusion Matrix for xgboost 
#iPhone
xgb.pred_i$prediction <- as.factor(xgb.pred_i$prediction)
xgb.pred_i$label <- as.factor(xgb.pred_i$label)
conf_mat_i_xgb<-confusionMatrix(xgb.pred_i$prediction, xgb.pred_i$label) 
conf_mat_i_xgb

save.image(file='Balanced data Iphone_Samsung sentiment')

stopCluster(cl)

