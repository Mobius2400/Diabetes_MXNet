## MXNet and Predicitve Analytics POC on Prime Indians Diabetes Database.
## Dataset source: https://www.kaggle.com/uciml/pima-indians-diabetes-database

## Load data
data <- read.csv("diabetes.csv")
## Split data into training and test with 70/30 ratio
require(caTools)
split <- sample.split(data, SplitRatio = 0.70)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

## Decision tree classification
require(rpart)
require(rattle)
require(rpart.plot)
require(RColorBrewer)
require(randomForest)
fit <- randomForest(as.factor(Outcome) ~ Pregnancies + Glucose + BloodPressure +
                      SkinThickness + Insulin + BMI + DiabetesPedigreeFunction +
                      Age, data=train, importance=TRUE, ntree=2000)
fancyRpartPlot(fit)

## Predict using Decision Tree
Prediction <- predict(fit, test, type = "class")
## Calculate model accuracy
classes <- table(Prediction, test$Outcome)
accuracy <- ((classes["0","0"]+classes["1","1"])/nrow(test))*100
cat("The accuracy of the model is: ", accuracy)