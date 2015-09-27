library(caret)

set.seed(3141)

baseTrain <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
submissionSet <- read.csv("pml-testing.csv", stringsAsFactors = FALSE)

baseTrain$classe <- factor(baseTrain$classe)

#variable X and timestamps should be removed from analysis as irrelevant to classification
baseTrain <- baseTrain[-c(1:7)]

fullTrain <- baseTrain[c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt",
 "gyros_belt_x", "gyros_belt_y", 
 "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", 
 "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", 
 "pitch_arm", "yaw_arm", "total_accel_arm", 
 "gyros_arm_x", "gyros_arm_y", "gyros_arm_z",
 "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x",
 "magnet_arm_y", "magnet_arm_z",
 "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
 "total_accel_dumbbell",
 "gyros_dumbbell_x", "gyros_dumbbell_y", 
 "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", 
 "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", 
 "pitch_forearm", "yaw_forearm", "total_accel_forearm", 
 "gyros_forearm_x", 
 "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", 
 "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", "classe")]

# coerce numeric variables to numeric
# cleanTrain <- data.frame(apply(baseTrain, 2, as.numeric))
# cleanTrain$classe <- baseTrain$classe

# train[apply(train[-153],2,sum)]

inTrain <- createDataPartition(fullTrain$classe, p = 0.7, list = FALSE)

train <- fullTrain[inTrain,]
test <- fullTrain[-inTrain,]

# Principal Components Analysis
# extract numeric variables for PCA

trainPC <- preProcess(train[-c(53)], method ="pca", thresh = 0.9) 
# predict(trainPC, train[-c(53)])
# train(train$classe ~ ., data = predict(trainPC, train[-c(53)]), method = "rpart")

# random forest
forestModel <- train(train$classe ~ ., data = predict(trainPC, train[-c(53)]), method = "rf")

#try gba tree boosting
# train(train$classe ~ ., data = predict(trainPC, train[-c(53)]), method = "gbm")

# try Linear Discrimiant Analysis
# train(train$classe ~ ., data = predict(trainPC, train[-c(53)]), method = "lda")

# validate model against test set
testPC <- predict(trainPC, test[-c(53)])
predictions <- predict(forestModel, newdata = testPC)

# test set accuracy
sum(test$classe==predictions)/length(predictions)


# prepare the submission set
submitPC <- predict(trainPC, submissionSet[c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt",
 "gyros_belt_x", "gyros_belt_y", 
 "gyros_belt_z", "accel_belt_x", "accel_belt_y", "accel_belt_z", 
 "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", 
 "pitch_arm", "yaw_arm", "total_accel_arm", 
 "gyros_arm_x", "gyros_arm_y", "gyros_arm_z",
 "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x",
 "magnet_arm_y", "magnet_arm_z",
 "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
 "total_accel_dumbbell",
 "gyros_dumbbell_x", "gyros_dumbbell_y", 
 "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", 
 "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", 
 "pitch_forearm", "yaw_forearm", "total_accel_forearm", 
 "gyros_forearm_x", 
 "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", 
 "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")])

submissionPredictions <- predict(forestModel, newdata = submitPC)

#write out predictions for submissionSet

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

setwd("./submit")
pml_write_files(submissionPredictions)
setwd("../")
