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

set.seed(123)

# recode sentiment to combine factor levels 1 negative 2 neutral 3 positive
#iPhone
iphoneRC <-iPhone
iphoneRC$iphonesentiment <- dplyr::recode(iPhone$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3) 

#Samsung
samsungRC <-samsung
samsungRC$galaxysentiment<- dplyr::recode(samsungRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 2, '4' = 3, '5' = 3) 

#Spliiting the data in to train and test set
#Iphone
n = nrow(iphoneRC)
train.index = sample(n,floor(0.7*n))
train.data_i = iphoneRC[train.index,]
test.data_i = iphoneRC[-train.index,]

#Samsung
train.data_s = samsungRC[train.index,]
test.data_s = samsungRC[-train.index,]

### Balancing train set
#iPhone
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

#Samsung
df1_s <- filter(train.data_s, galaxysentiment %in% c(1, 3))
df2_s <- filter(train.data_s, galaxysentiment %in% c(2, 3))
df3_s <- filter(train.data_s, galaxysentiment %in% c(3))

df1_over_s <- ovun.sample(galaxysentiment~., data = df1_s, method = "over", N = 12984)$data
table(df1_over_s$galaxysentiment)

df1_s <- filter(df1_over_s, galaxysentiment %in% c(1))

df2_over_s <- ovun.sample(galaxysentiment~., data = df2_s, method = "over", N = 12984)$data
table(df2_over_s$galaxysentiment)

df2_s <- filter(df2_over_s, galaxysentiment %in% c(2))
samsung_bal <- rbind(df1_s, df2_s, df3_s)

prop.table(table(samsung_bal$galaxysentiment))
barplot(prop.table(table(samsung_bal$galaxysentiment)),
        col = rainbow(4),
        ylim = c(0,0.5),
        main = "Class Distribution after ROSE")

train.data_s <- samsung_bal

#PCA
#iPhone
preprocessParams_i <- preProcess(train.data_i[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams_i)

#Samsung
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

large.pca_i <- predict(preprocessParams_i, large)
large.pca_s <- predict(preprocessParams_s, large)

# add the dependent to training
test.pca_i$iphonesentiment <- test.data_i$iphonesentiment
test.pca_s$galaxysentiment <- test.data_s$galaxysentiment

# inspect results
str(train.pca_s)
str(test.pca_s)
str(large.pca_s)

#Delete real values
#iPhone
train_iphonesentiment<- train.pca_i$iphonesentiment
test_iphonesentiment <- test.pca_i$iphonesentiment
train.pca_i$iphonesentiment <- NULL
test.pca_i$iphonesentiment <- NULL

#Samsung
train_galaxysentiment<- train.pca_s$galaxysentiment
test_galaxysentiment <- test.pca_s$galaxysentiment
train.pca_s$galaxysentiment <- NULL
test.pca_s$galaxysentiment <- NULL

#iPhone
train.data_i = as.matrix(train.pca_i)
train.label_i = as.integer(train_iphonesentiment)-1
test.data_i = as.matrix(test.pca_i)
test.label_i = as.integer(test_iphonesentiment)-1

#Samsung
train.data_s = as.matrix(train.pca_s)
train.label_s = as.integer(train_galaxysentiment)-1
test.data_s = as.matrix(test.pca_s)
test.label_s = as.integer(test_galaxysentiment)-1

#Large
large.data_i = as.matrix(large.pca_i)
large.data_s = as.matrix(large.pca_s)

# Transform the two data sets into xgb.Matrix
#iphone
xgb.train_i = xgb.DMatrix(data=train.data_i,label=train.label_i)
xgb.test_i = xgb.DMatrix(data=test.data_i,label=test.label_i)
#samsung
xgb.train_s = xgb.DMatrix(data=train.data_s,label=train.label_s)
xgb.test_s = xgb.DMatrix(data=test.data_s,label=test.label_s)

# Define the parameters for multinomial classification
#iPhone
iphonesentiment <-as.factor(iphoneRC$iphonesentiment)
galaxysentiment <- as.factor(samsungRC$galaxysentiment)
num_class_i = length(levels(iphonesentiment))

#Parameters for model
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
xgb.fit_i=xgb.train(
  params=params_i,
  data=xgb.train_i,
  nrounds=1000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train_i,val2=xgb.test_i),
  verbose=0
)

#samsung
xgb.fit_s=xgb.train(
  params=params_i,
  data=xgb.train_s,
  nrounds=1000,
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
#samsung
xgb.pred_s = predict(xgb.fit_s,test.data_s,reshape=T)
xgb.pred_s = as.data.frame(xgb.pred_s)
colnames(xgb.pred_s) = levels(iphonesentiment)

#large iPhone
xgb.pred_l_i = predict(xgb.fit_i,large.data_i,reshape=T)
xgb.pred_l_i = as.data.frame(xgb.pred_l_i)
colnames(xgb.pred_l_i) = levels(iphonesentiment)

xgb.pred_l_s = predict(xgb.fit_s,large.data_s,reshape=T)
xgb.pred_l_s = as.data.frame(xgb.pred_l_s)
colnames(xgb.pred_l_s) = levels(iphonesentiment)


# Use the predicted label with the highest probability

#iPhone
xgb.pred_i$prediction = apply(xgb.pred_i,1,function(x) colnames(xgb.pred_i)[which.max(x)])
xgb.pred_i$label = levels(iphonesentiment)[test.label_i+1]

#samsung
xgb.pred_s$prediction = apply(xgb.pred_s,1,function(x) colnames(xgb.pred_s)[which.max(x)])
xgb.pred_s$label = levels(iphonesentiment)[test.label_s+1]

# Calculate the final accuracy iPhone
Accuracy_i = sum(xgb.pred_i$prediction==xgb.pred_i$label)/nrow(xgb.pred_i)
print(paste("Final Accuracy iPhone =",sprintf("%1.2f%%", 100*Accuracy_i))) # Accuracy  82,12%, kappa: 0.5797
# Calculate the final accuracy samsung
Accuracy_s = sum(xgb.pred_s$prediction==xgb.pred_s$label)/nrow(xgb.pred_s)
print(paste("Final Accuracy iPhone =",sprintf("%1.2f%%", 100*Accuracy_s))) # Accuracy  84,22%, kappa: 0.6126

#Confusion Matrix for xgboost 
#iPhone
xgb.pred_i$prediction <- as.factor(xgb.pred_i$prediction)
xgb.pred_i$label <- as.factor(xgb.pred_i$label)
conf_mat_i_xgb<-confusionMatrix(xgb.pred_i$prediction, xgb.pred_i$label) 
conf_mat_i_xgb

#samsung
xgb.pred_s$prediction <- as.factor(xgb.pred_s$prediction)
xgb.pred_s$label <- as.factor(xgb.pred_s$label)
conf_mat_s_xgb<-confusionMatrix(xgb.pred_s$prediction, xgb.pred_s$label) 
conf_mat_s_xgb

#iPhone and Samsung
xgb.pred_l_i$prediction = apply(xgb.pred_l_i,1,function(x) colnames(xgb.pred_l_i)[which.max(x)])
xgb.pred_l_s$prediction = apply(xgb.pred_l_s,1,function(x) colnames(xgb.pred_l_s)[which.max(x)])

xgb.pred_l_i$prediction <- as.numeric(xgb.pred_l_i$prediction)
xgb.pred_l_s$prediction <- as.numeric(xgb.pred_l_s$prediction)

library(ggplot2)
library(ggpubr)
library(dplyr)

#iPhone
df_i <- xgb.pred_l_i %>%
  group_by(prediction) %>%
  summarise(counts = n())
df_i

#Samsung
df_s <- xgb.pred_l_s %>%
  group_by(prediction) %>%
  summarise(counts = n())
df_s

#Visualization of prediction

#iPhone seniment

ggplot(df_i, aes(x = prediction, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3) + 
  theme_pubclean()

df_i <- df_i %>%
  arrange(desc(prediction)) %>%
  mutate(prop = round(counts*100/sum(counts), 1),
         lab.ypos = cumsum(prop) - 0.5*prop)
head(df_i, 4)
df_i$prediction<- as.factor(df_i$prediction)

ggplot(df_i, aes(x = "", y = prop, fill = prediction)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  geom_text(aes(y = lab.ypos, label = prop), color = "white")+
  coord_polar("y", start = 0)+
  ggpubr::fill_palette("jco")+
  theme_void()

#Samsung sentiment


ggplot(df_s, aes(x = prediction, y = counts)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = counts), vjust = -0.3) + 
  theme_pubclean()

df_s <- df_s %>%
  arrange(desc(prediction)) %>%
  mutate(prop = round(counts*100/sum(counts), 1),
         lab.ypos = cumsum(prop) - 0.5*prop)
head(df_s, 4)
df_s$prediction<- as.factor(df_s$prediction)

ggplot(df_s, aes(x = "", y = prop, fill = prediction)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  geom_text(aes(y = lab.ypos, label = prop), color = "white")+
  coord_polar("y", start = 0)+
  ggpubr::fill_palette("jco")+
  theme_void()

save.image(file='Balanced data Iphone_Samsung sentiment')

stopCluster(cl)

