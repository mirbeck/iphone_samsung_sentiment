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

# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iPhone$iphonesentiment <- dplyr::recode(iPhone$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 

#Examine classes proportion
prop.table(table(iPhone$iphonesentiment))
barplot(prop.table(table(iPhone$iphonesentiment)),
        col = rainbow(4),
        ylim = c(0,0.7),
        main = "Class Distribution")

df1 <- filter(iPhone, iphonesentiment %in% c(1, 4))
df2 <- filter(iPhone, iphonesentiment %in% c(2, 4))
df3 <- filter(iPhone, iphonesentiment %in% c(3, 4))
df4 <- filter(iPhone, iphonesentiment %in% c(4))

library(ROSE)

df1_over <- ovun.sample(iphonesentiment~., data = df1, method = "over", N = 17958)$data
table(df1_over$iphonesentiment)

df1 <- filter(df1_over, iphonesentiment %in% c(1))

df2_over <- ovun.sample(iphonesentiment~., data = df2, method = "over", N = 17958)$data
table(df2_over$iphonesentiment)

df2 <- filter(df2_over, iphonesentiment %in% c(2))

df3_over <- ovun.sample(iphonesentiment~., data = df3, method = "over", N = 17958)$data
table(df3_over$iphonesentiment)

df3 <- filter(df3_over, iphonesentiment %in% c(3))

iphone_bal <- rbind(df1, df2, df3, df4)

prop.table(table(iphone_bal$iphonesentiment))
barplot(prop.table(table(iphone_bal$iphonesentiment)),
        col = rainbow(4),
        ylim = c(0,0.5),
        main = "Class Distribution after ROSE")

write.csv(iphone_bal, "iphone_bal.csv")
