## MXNet and Predicitve Analytics POC on Prime Indians Diabetes Database.
## Dataset source: https://www.kaggle.com/uciml/pima-indians-diabetes-database

## Load data
data <- read.csv("diabetes.csv")
## Split data into training and test with 70/30 ratio
require(caTools)
split <- sample.split(data, SplitRatio = 0.70)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

## Process train dataset to extract parameters and labels.
train_label <- train[,ncol(train)]
train_data <- train[c(1:ncol(train)-1)]
## Normalize training data.
train_data <- scale(train_data)

## Create network model and train using MXNet:
require(mxnet)
mx.set.seed(101)
model <- mx.mlp(data.matrix(train_data), train_label, 
                hidden_node=ncol(train_data), out_node=1, 
                out_activation="rmse",
                num.round=25, array.batch.size=15, learning.rate=0.07, momentum=0.9,
                eval.metric=mx.metric.rmse)

## Process test dataset to extract parameters and labels.
test_label <- test[,ncol(test)]
test_data <- test[c(1:ncol(test)-1)]
## Normalize training data.
test_data <- scale(test_data)

## Predict outcome using developed model:
predictions <- predict(model, data.matrix(test_data))
predictions <- round(predictions)
## Calculate model accuracy

cat("The accuracy of the model is: ")
