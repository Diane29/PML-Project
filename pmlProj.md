# Practical Machine Learning Final Project
D. Jones  
October 28, 2016  



#Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.
In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

We will predict the manner in which the individuals did the exercises using the "classe" variable in the training set. A report is then created describing how the model is built, how it was cross-validated, the expected out of sample error, and reasons for the decisions made in the project. The prediction model will then be used to predict 20 different test cases. The following describes the classes of exercises to be predicted in the models using the variable "classe":
  .Class A: exactly according to the specification
  .Class B: throwing the elbows to the front
  .Class C: lifting the dumbbell only halfway
  .Class D: lowering the dumbbell only half way
  .Class E: throwing the hips to the front.


#Loading the Data

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.4
```

```r
rm(list = ls())
if (!file.exists("pml-training.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
}
pml_train <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA"))  
pml_test <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA"))
dim(pml_train)
```

```
## [1] 19622   160
```

```r
dim(pml_test)
```

```
## [1]  20 160
```

```r
#colnames(pml_train)==colnames(pml_test)
```
The datasets consist of 160 variables. The data shows that many of the variables are not informative and will be deleted from the datasets. Also, check to make sure the Training and Testing dataset has the same columns.

#Processing the Training Dataset

```r
features <- names(pml_train[,colSums(is.na(pml_train)) == 0])
featTest <- names(pml_test[,colSums(is.na(pml_test)) == 0])
pmlTrain <- pml_train[,features]
pmlTest <- pml_test[,featTest]
pmlMod <-  pmlTrain[, -c(1:7)]
pmlTest <-  pmlTest[, -c(1:7)]
dim(pmlMod); dim(pmlTest)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

```r
#colnames(pmlMod)
```
The variables that do not have NA values are used as the predictors in the model. An additional 7 variables are removed from the datasets that would not be useful in explaining the model such as username and the timestamp variables. The dataset now contains 52 variables to be modeled against the "classe" variable.


#Partitioning the Training Dataset

```r
set.seed(4631) 
inTrain <- createDataPartition(pmlMod$classe, p = 0.6, list = FALSE)
pmlModTrain <- pmlMod[inTrain, ]
pmlModValid <- pmlMod[-inTrain, ]
dim(pmlModTrain); dim(pmlModValid)
```

```
## [1] 11776    53
```

```
## [1] 7846   53
```
The training dataset is partitioned into 60/40 split in order to have a decent dataset to do the cross-validation. 

#Prediction using Rpart Decision Tree

```r
library(rattle); library(rpart); library(rpart.plot)
```

```
## Warning: package 'rattle' was built under R version 3.2.5
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.5
```

```r
trainCtrl<- trainControl(method="repeatedcv", number=3)
modFitpml <- train(classe ~ ., trControl=trainCtrl, preProc = c("center", "scale"),method="rpart",data =pmlModTrain)

par(mar=c(3.1,3.1,3.1,3.1))
fancyRpartPlot(modFitpml$finalModel)
```

![](pmlProj_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
predClass <- predict(modFitpml,newdata=pmlModValid)
confusionMatrix(pmlModValid$classe,predClass)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1992   37  172    0   31
##          B  627  503  388    0    0
##          C  620   48  700    0    0
##          D  562  248  476    0    0
##          E  191  179  373    0  699
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4963          
##                  95% CI : (0.4852, 0.5074)
##     No Information Rate : 0.5088          
##     P-Value [Acc > NIR] : 0.9869          
##                                           
##                   Kappa : 0.3427          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4990  0.49557  0.33191       NA  0.95753
## Specificity            0.9377  0.85141  0.88356   0.8361  0.89559
## Pos Pred Value         0.8925  0.33136  0.51170       NA  0.48474
## Neg Pred Value         0.6437  0.91909  0.78249       NA  0.99516
## Prevalence             0.5088  0.12937  0.26880   0.0000  0.09304
## Detection Rate         0.2539  0.06411  0.08922   0.0000  0.08909
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7184  0.67349  0.60774       NA  0.92656
```

The Rpart function is used for the initial model with a repeated 10-fold cross-validation. The accuracy of the model is 49.6% with an expected out-of-sample error rate of 50.4%. This means that the model is not very good. A Random Forest model will be used instead because it uses multiple models for better performance.

#Prediction using Random Forest

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
modFitpmlRF <- randomForest(classe ~. ,data=pmlModTrain, method="class")
predictRF <- predict(modFitpmlRF, pmlModValid, type="class")
predTest <- predict(modFitpmlRF, pmlTest, type = "class")
confusionMatrix(predictRF, pmlModValid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   13    0    0    0
##          B    2 1495    5    0    0
##          C    0   10 1363   17    0
##          D    0    0    0 1267    4
##          E    1    0    0    2 1438
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9931         
##                  95% CI : (0.991, 0.9948)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9913         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9848   0.9963   0.9852   0.9972
## Specificity            0.9977   0.9989   0.9958   0.9994   0.9995
## Pos Pred Value         0.9942   0.9953   0.9806   0.9969   0.9979
## Neg Pred Value         0.9995   0.9964   0.9992   0.9971   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1905   0.1737   0.1615   0.1833
## Detection Prevalence   0.2858   0.1914   0.1772   0.1620   0.1837
## Balanced Accuracy      0.9982   0.9919   0.9961   0.9923   0.9984
```

```r
plot(modFitpmlRF)
```

![](pmlProj_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

The random forest model gives an accuracy rate of 99.3% using all the 53 variables in the testing dataset. This results in an expected out-of-sample error rate of 0.7% indicating a very good model. The plot of the model fit shows that as the number of trees increase the error rate decreases. The model requires a lot of computationtal time so this is not the most efficient model. However, the random forest method reduces overfitting and is good for nonlinear features. The model is used on the test dataset to predict the 20 cases.

