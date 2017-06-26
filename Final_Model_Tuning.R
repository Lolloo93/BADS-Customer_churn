set.seed(2)
k <-5 

# Create training and test sets

idx.train <- createDataPartition(y= df_known3$return_customer, p=0.8, list = FALSE)
train <- df_known3[idx.train,]
test <- df_known3[-idx.train,]

train.rnd <- sample(nrow(train))
folds <- cut(1:nrow(train), breaks = k, labels = FALSE)


# Selecting the folds by creating a list of k sets of indices # 

folds.trainrowindices <- lapply(1:k, function(x) train.rnd[which(folds!=x)])
folds.validationrowindices <- lapply(folds.trainrowindices, function(x) setdiff(1:nrow(train),x))




### Random Forest ###

if(!require("Rborist")) install.packages("Rborist"); library("Rborist")

# Define the parameter grid #

rf.parGrid <- expand.grid(predFixed=seq(3,11,2), ntree = c(250,500,750,1000,1500)) 

model.control <- trainControl(
  method = "none", # manual cv later in the loop
  #number = 2, # number of folds in cross validation
  classProbs = TRUE, # Return class probabilities
  #summaryFunction = twoClassSummary, # twoClassSummary returns AUC
  allowParallel = TRUE # Enable parallelization if available
)

# Set up the foreach-loops: #

# Outside loop: #

results.rf <- foreach(i = 1:nrow(rf.parGrid), .combine = rbind, .packages = c("caret", "Rborist", "pROC"))%:%
  
  # Inside loop: #
  
  foreach(j=1:k, .combine=c, .packages = c("caret", "Rborist", "pROC")) %dopar% {  # parallel comp
  
    cv.train <- train[folds.trainrowindices[[j]],]
    cv.val <- train[folds.validationrowindices[[j]],]
    
    
    rf <- train(x= cv.train[,-26], y=cv.train[,26], trControl = model.control, method="Rborist",
                        tuneGrid = expand.grid(predFixed = rf.parGrid[i], ntree = rf.parGrid$ntree[i]))
    yhat <- predict(rf, newdata= cv.val, type ="prob")[,2]
    gain <- evaluation.costMatrix1(yhat, as.numeric(cv.val$return_customer)-1)

    return(gain$percentage_potential)
  }

# Calculate fold average #

results.rf.foldAverage <- rowMeans(results.rf)

idx.rf.besttuned <- which.max(results.rf.foldAverage) # model with best parameter combination

# train randomforest again with best parameter combination:

rf.final <- train(return_customer~., data = train, trControl = model.control, method = "Rborist",
                          ntree = rf.parGrid$ntree[idx.rf.besttuned], tuneGrid = expand.grid(predFixed = rf.parGrid$predFixed[idx.rf.besttuned]))

# predict on test set:

yhat.rf <- predict(rf.final, newdata = test, type = "prob")[,2]

gain.rf <- evaluation.costMatrix1(yhat.rf, as.numeric(test$return_customer)-1)

########################################################################################################

### Neural Network ###

if(!require("nnet")) install.packages("nnet"); library("nnet")

# Define the parameter grid #

nn.parGrid <- expand.grid(size = seq(3,11,2), decay = c(0.001,0.005,0.01,0.05,0.1)) 


# Set up the foreach-loops: #

# Outside loop: #

results.nn <- foreach(i = 1:nrow(nn.parGrid), .combine = rbind, .packages = c("caret", "nnet", "pROC"))%:%
  
  # Inside loop: #
  
  foreach(j=1:k, .combine=c, .packages = c("caret", "nnet", "pROC")) %dopar% {  # parallel comp
    
    cv.train <- train[folds.trainrowindices[[j]],]
    cv.val <- train[folds.validationrowindices[[j]],]
    
    
    nn <- train(x= cv.train[,-26], y=cv.train[,26], trControl = model.control, method="nnet",
                tuneGrid = expand.grid(size = nn.parGrid$size[i], decay = nn.parGrid$decay[i]), maxit = 1000)
    yhat <- predict(nn, newdata= cv.val, type ="prob")[,2]
    gain <- evaluation.costMatrix1(yhat, as.numeric(cv.val$return_customer)-1)
    
    return(gain$percentage_potential)
  }

# Calculate fold average #

results.nn.foldAverage <- rowMeans(results.nn)

idx.nn.besttuned <- which.max(results.nn.foldAverage) # model with best parameter combination

# train randomforest again with best parameter combination:

nn.final <- train(return_customer~., data = train, trControl = model.control, method = "nnet",
                  tuneGrid = expand.grid(decay = nn.parGrid$decay[idx.nn.besttuned], size = nn.parGrid$decay[idx.nn.besttuned]))

# predict on test set:

yhat.nn <- predict(nn.final, newdata = test, type = "prob")[,2]

gain.nn <- evaluation.costMatrix1(yhat.nn, as.numeric(test$return_customer)-1)

###########################################################################################################

### Gradient Boosting ###

if(!require("xgboost")) install.packages("xgboost"); library("xgboost")

# Define the parameter grid #

xgb.parGrid <- expand.grid(nrounds = c(100,500,1500),
                           max_depth = c(3,7,10),
                           eta = c(0.001,0.005, 0.01),
                           gamma = 0,
                           colsample_bytree = c(0.5,0.8),
                           min_child_weight = 1,
                           subsample = c(0.8,1))


# Set up the foreach-loops: #

# Outside loop: #

results.xgb <- foreach(i = 1:nrow(xgb.parGrid), .combine = rbind, .packages = c("caret", "xgboost", "pROC"))%:%
  
  # Inside loop: #
  
  foreach(j=1:k, .combine=c, .packages = c("caret", "xgboost", "pROC")) %dopar% {  # parallel comp
   
    cv.train <- train[folds.trainrowindices[[j]],]
    cv.val <- train[folds.validationrowindices[[j]],]
    
    
    xgb <- train(return_customer~., data = train, trControl = model.control, method = "xgbTree",
                 tuneGrid = expand.grid(nrounds=xgb.parGrid$nrounds[i], max_depth = xgb.parGrid$max_depth[i],
                                        eta = xgb.parGrid$eta[i], gamma = xgb.parGrid$gamma[i],
                                        colsample_bytree = xgb.parGrid$colsample_bytree[i], min_child_weight = xgb.parGrid$min_child_weight[i],
                                        subsample = xgb.parGrid$subsample[i]))
    yhat <- predict(xgb, newdata= cv.val, type ="prob")[,2]
    gain <- evaluation.costMatrix1(yhat, as.numeric(cv.val$return_customer)-1)
    
    return(gain$percentage_potential)
  }

# Calculate fold average #

results.xgb.foldAverage <- rowMeans(results.xgb)

idx.xgb.besttuned <- which.max(results.xgb.foldAverage) # model with best parameter combination

# train randomforest again with best parameter combination:

xgb.final <- train(return_customer~., data = train, trControl = model.control, method = "xgbTree",
                  tuneGrid = expand.grid(nrounds=xgb.parGrid$nrounds[idx.xgb.besttuned], max_depth = xgb.parGrid$max_depth[idx.xgb.besttuned],
                                         eta = xgb.parGrid$eta[idx.xgb.besttuned], gamma = xgb.parGrid$gamma[idx.xgb.besttuned],
                                         colsample_bytree = xgb.parGrid$colsample_bytree[idx.xgb.besttuned], min_child_weight = xgb.parGrid$min_child_weight[idx.xgb.besttuned],
                                         subsample = xgb.parGrid$subsample[idx.xgb.besttuned]))

# predict on test set:

yhat.xgb <- predict(xgb.final, newdata = test, type = "prob")[,2]

gain.xgb <- evaluation.costMatrix1(yhat.xgb, as.numeric(test$return_customer)-1)

############################################################################################################

### Logistic Regression ###

# No parametr tuning and no cross validation needed # 

lr <- train(return_customer~., data = train, method = "glm", family = "binomial")

yhat.lr <- predict(lr, newdata = test, type = "prob")[,2]

gain.lr <- evaluation.costMatrix1(yhat.lr, as.numeric(test$return_customer)-1)


### Put single model results together in a table to compare performance 

basemodel.comp <- cbind(gain.rf$percentage_potential, gain.nn$percentage_potential, gain.xgb$percentage_potential,
                        gain.lr$percentage_potential)

#############################################################################################################

### Repeat the above steps on SMOTE data ### 

if(!require("DMwR")) install.packages("DMwR"); library("DMwR")
if(!require("FNN")) install.packages("FNN"); library("FNN")

set.seed(1)
k <-5 

# Create training and test sets

idx.train <- createDataPartition(y= df_known3$return_customer, p=0.8, list = FALSE)
train <- df_known3[idx.train,]
test <- df_known3[-idx.train,]

train.smote <- SMOTE(return_customer~., data = train, k= 5, perc.over = 100, perc.under = 200)

train.smote.rnd <- sample(nrow(train.smote))
folds.smote <- cut(1:nrow(train.smote), breaks = k, labels = FALSE)


# Selecting the folds by creating a list of k sets of indices # 

folds.trainrowindices <- lapply(1:k, function(x) train.smote.rnd[which(folds.smote!=x)])
folds.validationrowindices <- lapply(folds.trainrowindices, function(x) setdiff(1:nrow(train.smote),x))


if(!require("Rborist")) install.packages("Rborist"); library("Rborist")

# Define the parameter grid #

rf.parGrid <- expand.grid(predFixed=seq(3,11,2), ntree = c(250,500,750,1000,1500)) 


# Set up the foreach-loops: #

# Outside loop: #

results.rf.smote <- foreach(i = 1:nrow(rf.parGrid), .combine = rbind, .packages = c("caret", "Rborist", "pROC"))%:%
  
  # Inside loop: #
  
  foreach(j=1:k, .combine=c, .packages = c("caret", "Rborist", "pROC")) %dopar% {  # parallel comp
    
    cv.train <- train.smote[folds.trainrowindices[[j]],]
    cv.val <- train.smote[folds.validationrowindices[[j]],]
    
    
    rf <- train(x= cv.train[,-26], y=cv.train[,26], trControl = model.control, method="Rborist",
                tuneGrid = expand.grid(predFixed = rf.parGrid[i], ntree = rf.parGrid$ntree[i]))
    yhat <- predict(rf, newdata= cv.val, type ="prob")[,2]
    gain <- evaluation.costMatrix1(yhat, as.numeric(cv.val$return_customer)-1)
    
    return(gain$percentage_potential)
  }

# Calculate fold average #

results.rf.smote.foldAverage <- rowMeans(results.rf.smote)

idx.rf.besttuned <- which.max(results.rf.smote.foldAverage) # model with best parameter combination

# train randomforest again with best parameter combination:

rf.smote.final <- train(return_customer~., data = train.smote, trControl = model.control, method = "Rborist",
                  ntree = rf.parGrid$ntree[idx.rf.besttuned], tuneGrid = expand.grid(predFixed = rf.parGrid$predFixed[idx.rf.besttuned]))

# predict on test set:

yhat.rf.smote <- predict(rf.smote.final, newdata = test, type = "prob")[,2]

gain.rf.smote <- evaluation.costMatrix1(yhat.rf.smote, as.numeric(test$return_customer)-1)

########################################################################################################

### Neural Network ###

if(!require("nnet")) install.packages("nnet"); library("nnet")

# Define the parameter grid #

nn.parGrid <- expand.grid(size = seq(3,11,2), decay = c(0.001,0.005,0.01,0.05,0.1)) 

# Set up the foreach-loops: #

# Outside loop: #

results.nn.smote <- foreach(i = 1:nrow(nn.parGrid), .combine = rbind, .packages = c("caret", "nnet", "pROC"))%:%
  
  # Inside loop: #
  
  foreach(j=1:k, .combine=c, .packages = c("caret", "nnet", "pROC")) %dopar% {  # parallel comp
    
    cv.train <- train.smote[folds.trainrowindices[[j]],]
    cv.val <- train.smote[folds.validationrowindices[[j]],]
    
    
    nn <- train(x= cv.train[,-26], y=cv.train[,26], trControl = model.control, method="nnet",
                tuneGrid = expand.grid(size = nn.parGrid$size[i], decay = nn.parGrid$decay[i]), maxit = 1000)
    yhat <- predict(nn, newdata= cv.val, type ="prob")[,2]
    gain <- evaluation.costMatrix1(yhat, as.numeric(cv.val$return_customer)-1)
    
    return(gain$percentage_potential)
  }

# Calculate fold average #

results.nn.smote.foldAverage <- rowMeans(results.nn.smote)

idx.nn.besttuned <- which.max(results.nn.smote.foldAverage) # model with best parameter combination

# train again with best parameter combination:

nn.final.smote <- train(return_customer~., data = train.smote, trControl = model.control, method = "nnet",
                  tuneGrid = expand.grid(decay = nn.parGrid$decay[idx.nn.besttuned], size = nn.parGrid$decay[idx.nn.besttuned]))

# predict on test set:

yhat.nn.smote <- predict(nn.final.smote, newdata = test, type = "prob")[,2]

gain.nn.smote <- evaluation.costMatrix1(yhat.nn.smote, as.numeric(test$return_customer)-1)

###########################################################################################################

### Gradient Boosting ###

if(!require("xgboost")) install.packages("xgboost"); library("xgboost")

# Define the parameter grid #

xgb.parGrid <- expand.grid(nrounds = c(100,500,1500),
                           max_depth = c(3,7,10),
                           eta = c(0.001,0.005, 0.01),
                           gamma = 0,
                           colsample_bytree = c(0.5,0.8),
                           min_child_weight = 1,
                           subsample = c(0.8,1))


# Set up the foreach-loops: #

# Outside loop: #

results.xgb.smote <- foreach(i = 1:nrow(xgb.parGrid), .combine = rbind, .packages = c("caret", "xgboost", "pROC"))%:%
  
  # Inside loop: #
  
  foreach(j=1:k, .combine=c, .packages = c("caret", "xgboost", "pROC")) %dopar% {  # parallel comp
    
    cv.train <- train.smote[folds.trainrowindices[[j]],]
    cv.val <- train.smote[folds.validationrowindices[[j]],]
    
    
    xgb <- train(return_customer~., data = cv.train, trControl = model.control, method = "xgbTree",
                 tuneGrid = expand.grid(nrounds=xgb.parGrid$nrounds[i], max_depth = xgb.parGrid$max_depth[i],
                                        eta = xgb.parGrid$eta[i], gamma = xgb.parGrid$gamma[i],
                                        colsample_bytree = xgb.parGrid$colsample_bytree[i], min_child_weight = xgb.parGrid$min_child_weight[i],
                                        subsample = xgb.parGrid$subsample[i]))
    yhat <- predict(xgb, newdata= cv.val, type ="prob")[,2]
    gain <- evaluation.costMatrix1(yhat, as.numeric(cv.val$return_customer)-1)
    
    return(gain$percentage_potential)
  }

# Calculate fold average #

results.xgb.smote.foldAverage <- rowMeans(results.xgb.smote)

idx.xgb.smote.besttuned <- which.max(results.xgb.smote.foldAverage) # model with best parameter combination

# train randomforest again with best parameter combination:

xgb.final.smote <- train(return_customer~., data = train.smote, trControl = model.control, method = "xgbTree",
                   tuneGrid = expand.grid(nrounds=xgb.parGrid$nrounds[idx.xgb.besttuned], max_depth = xgb.parGrid$max_depth[idx.xgb.besttuned],
                                          eta = xgb.parGrid$eta[idx.xgb.besttuned], gamma = xgb.parGrid$gamma[idx.xgb.besttuned],
                                          colsample_bytree = xgb.parGrid$colsample_bytree[idx.xgb.besttuned], min_child_weight = xgb.parGrid$min_child_weight[idx.xgb.besttuned],
                                          subsample = xgb.parGrid$subsample[idx.xgb.besttuned]))

# predict on test set:

yhat.xgb.smote <- predict(xgb.final.smote, newdata = test, type = "prob")[,2]

gain.xgb.smote <- evaluation.costMatrix1(yhat.xgb.smote, as.numeric(test$return_customer)-1)

############################################################################################################

### Logistic Regression ###

# No parametr tuning and no cross validation needed # 

lr.smote <- train(return_customer~., data = train.smote, method = "glm", family = "binomial")

yhat.lr.smote <- predict(lr.smote, newdata = test, type = "prob")[,2]

gain.lr.smote <- evaluation.costMatrix1(yhat.lr.smote, as.numeric(test$return_customer)-1)


### Put single model results together in a table to compare performance 

smotemodel.comp <- cbind(gain.rf.smote$percentage_potential, gain.nn.smote$percentage_potential, gain.xgb.smote$percentage_potential,
                        gain.lr.smote$percentage_potential)

finalcomp <- rbind(basemodel.comp,smotemodel.comp)

###########################################################################################################

###### Buidling Heterogenous Ensembles by Stacking ######

### First base models on regular data ###

if(!require("caretEnsemble")) install.packages("caretEnsemble"); library("caretEnsemble")
if(!require("extraTrees")) install.packages("extraTrees"); library("extraTrees")

# 1.) Configure training process

control.ensemble <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = "final",
                                  summaryFunction = twoClassSummary)


# 2.) Choose the models to be used for the ensemble: 

modelList.ensemble  <- list(caretModelSpec(method = "Rborist", ntree=1000, ntree = rf.parGrid$ntree[idx.rf.besttuned], tuneGrid = expand.grid(predFixed = rf.parGrid$predFixed[idx.rf.besttuned]), metric="ROC"),
                            caretModelSpec(method = "glm", family="binomial", metric="ROC"),
                            caretModelSpec(method = "nnet", tuneGrid = expand.grid(decay = nn.parGrid$decay[idx.nn.besttuned], size = nn.parGrid$decay[idx.nn.besttuned]), maxit=1000, metric="ROC"),
                            caretModelSpec(method = "xgbTree", tuneGrid = expand.grid(nrounds=xgb.parGrid$nrounds[idx.xgb.besttuned], max_depth = xgb.parGrid$max_depth[idx.xgb.besttuned],
                                                                                      eta = xgb.parGrid$eta[idx.xgb.besttuned], gamma = xgb.parGrid$gamma[idx.xgb.besttuned],
                                                                                      colsample_bytree = xgb.parGrid$colsample_bytree[idx.xgb.besttuned], min_child_weight = xgb.parGrid$min_child_weight[idx.xgb.besttuned],
                                                                                      subsample = xgb.parGrid$subsample[idx.xgb.besttuned]), metric="ROC"))


# 3.) Collect the base model predictions: 

ensemble.modelPredictions <- caretList(return_customer~., data = train, trControl = control.ensemble,
                                        tuneList = modelList.ensemble, continue_on_fail = FALSE)

### Extract the model predicitions to further investigate the similarity and correlation between the base models
### -> aim is to have a set of base models that diverge regarding their predictions

pred.matrix<- cbind(ensemble.modelPredictions$Rborist$pred[,5], ensemble.modelPredictions$xgbTree$pred[,11],
                    ensemble.modelPredictions$glm$pred[,5], ensemble.modelPredictions$nnet$pred[,5])

if(!require("corrplot")) install.packages("corrplot"); library("corrplot")

cor.pres <- cor(pred.matrix)

corrplot(cor.pres, method = "circle")

# 4.) Use xgb as stacking classifier: 


ensemble.stacking.control <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                                           summaryFunction = twoClassSummary)

ensemble.stacking.model <- caretStack(ensemble.modelPredictions, method = "xgbTree",
                                       metric="ROC", trControl=ensemble.stacking.control,
                                       tuneGrid=expand.grid(nrounds=xgb.parGrid$nrounds[idx.xgb.besttuned], max_depth = xgb.parGrid$max_depth[idx.xgb.besttuned],
                                                            eta = xgb.parGrid$eta[idx.xgb.besttuned], gamma = xgb.parGrid$gamma[idx.xgb.besttuned],
                                                            colsample_bytree = xgb.parGrid$colsample_bytree[idx.xgb.besttuned], min_child_weight = xgb.parGrid$min_child_weight[idx.xgb.besttuned],
                                                            subsample = xgb.parGrid$subsample[idx.xgb.besttuned]))  ###### if tuning again, foreach with gain

# 5.) Predict test set and calculate AUC/ROC: 

ensemble.stacking.predictions <- predict(ensemble.stacking.model, newdata = test, type="prob")
roc.ensemble <- roc(test$return_customer, ensemble.stacking.predictions)
plot(roc.ensemble)

gain.ensemble <- evaluation.costMatrix1(ensemble.stacking.predictions, as.numeric(test$return_customer)-1)


######### Now manipulating the base models

# won't iterate through a big variety, no time anymore
# we will anyway say that we tried different combinations with different weights and found that to be optimal
# will do this in the analysis part

modelList.ensemble.opt  <- list(caretModelSpec(method = "Rborist", ntree = rf.parGrid$ntree[idx.rf.besttuned], tuneGrid = expand.grid(predFixed = rf.parGrid$predFixed[idx.rf.besttuned]), metric="ROC"),
                            caretModelSpec(method = "Rborist", ntree = rf.parGrid$ntree[idx.rf.besttuned], tuneGrid = expand.grid(predFixed = rf.parGrid$predFixed[idx.rf.besttuned]), classWeight=c(0.5,1.75), metric= "ROC"),
                            caretModelSpec(method = "Rborist", ntree = rf.parGrid$ntree[idx.rf.besttuned], tuneGrid = expand.grid(predFixed = rf.parGrid$predFixed[idx.rf.besttuned]), classWeight="balance", metric= "ROC"),
                            caretModelSpec(method = "glm", family="binomial", metric="ROC"),
                            caretModelSpec(method = "nnet", tuneGrid = expand.grid(decay = nn.parGrid$decay[idx.nn.besttuned], size = nn.parGrid$decay[idx.nn.besttuned]), maxit=1000, metric="ROC"),
                            caretModelSpec(method = "xgbTree", tuneGrid = expand.grid(nrounds=xgb.parGrid$nrounds[idx.xgb.besttuned], max_depth = xgb.parGrid$max_depth[idx.xgb.besttuned],
                                                                                      eta = xgb.parGrid$eta[idx.xgb.besttuned], gamma = xgb.parGrid$gamma[idx.xgb.besttuned],
                                                                                      colsample_bytree = xgb.parGrid$colsample_bytree[idx.xgb.besttuned], min_child_weight = xgb.parGrid$min_child_weight[idx.xgb.besttuned],
                                                                                      subsample = xgb.parGrid$subsample[idx.xgb.besttuned]), metric="ROC"))


# 3.) Collect the base model predictions: 

ensemble.opt.modelPredictions <- caretList(return_customer~., data = train, trControl = control.ensemble,
                                       tuneList = modelList.ensemble.opt, continue_on_fail = FALSE)


# 4.) Use xgb as stacking classifier: 


ensemble.stacking.control <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                                          summaryFunction = twoClassSummary)

ensemble.stacking.model.opt <- caretStack(ensemble.opt.modelPredictions, method = "xgbTree",
                                      metric="ROC", trControl=ensemble.stacking.control,
                                      tuneGrid=expand.grid(nrounds=xgb.parGrid$nrounds[idx.xgb.besttuned], max_depth = xgb.parGrid$max_depth[idx.xgb.besttuned],
                                                           eta = xgb.parGrid$eta[idx.xgb.besttuned], gamma = xgb.parGrid$gamma[idx.xgb.besttuned],
                                                           colsample_bytree = xgb.parGrid$colsample_bytree[idx.xgb.besttuned], min_child_weight = xgb.parGrid$min_child_weight[idx.xgb.besttuned],
                                                           subsample = xgb.parGrid$subsample[idx.xgb.besttuned]))  ###### if tuning again, foreach with gain

# 5.) Predict test set and calculate AUC/ROC: 

ensemble.stacking.opt.predictions <- predict(ensemble.stacking.model.opt, newdata = test, type="prob")
roc.ensemble <- roc(test$return_customer, ensemble.stacking.opt.predictions)
plot(roc.ensemble)

gain.ensemble.opt <- evaluation.costMatrix1(ensemble.stacking.opt.predictions, as.numeric(test$return_customer)-1)





