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
## Process test dataset to extract parameters and labels.
test_label <- test[,ncol(test)]
test_data <- test[c(1:ncol(test)-1)]
## Normalize training data.
test_data <- scale(test_data)

## Load model back
model = mx.model.FeedForward.load("diabetes_model", 100)
## Predict outcome using developed model:
predictions <- predict(model, data.matrix(test_data))
predictions <- round(predictions[2,])
## Calculate model accuracy
classes <- table(predictions, test_label)
accuracy <- ((classes["0","0"]+classes["1","1"])/nrow(test_data))*100
cat("The accuracy of the model is: ", accuracy)