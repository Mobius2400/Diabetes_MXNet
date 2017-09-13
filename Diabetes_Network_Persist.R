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

## Create network model and train using MXNet:
require(mxnet)
mx.set.seed(101)
# configure a two layer neuralnetwork
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type='relu')
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=64)
softmax = mx.symbol.SoftmaxOutput(fc2, name='sm')
model <- mx.model.FeedForward.create(symbol=softmax,
         X=data.matrix(train_data),y=train_label,
         ctx=mx.cpu(0), num.round=50, learning.rate=0.7,
         momentum=0.9, eval.metric=mx.metric.rmse,
         wd=0.001, epoch.end.callback=mx.callback.save.checkpoint("diabetes_model"),
         batch.end.callback=mx.callback.log.speedometer(batch.size=20, frequency = 100))
## Predict outcome using developed model:
predictions <- predict(model, data.matrix(test_data))
predictions <- round(predictions[2,])
## Calculate model accuracy
classes <- table(predictions, test_label)
accuracy <- ((classes["0","0"]+classes["1","1"])/nrow(test_data))*100
cat("The accuracy of the model is: ", accuracy)