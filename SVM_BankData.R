library(dummies)
library(rpart)
library(e1071)
library(caret)
library(rpart.plot)
library(rattle)
library(ggplot2)
library(C50)


set.seed(100)

## DATA IMPORT
bankdata = read.csv(file = "../bank-full.csv", header = TRUE, sep = ';')

## SPLIT DATA INTO TRAINING AND TESTING
bankdatasample <- sample.int(n = nrow(bankdata), size = floor(.70*nrow(bankdata)), replace = F)
bank_traindata <- bankdata[bankdatasample, ]
bank_testdata  <- bankdata[-bankdatasample, ]


#fit2 <- rpart(y ~ ., data = bank_traindata, parms = list(prior = c(.65,.35)),split = "information")

######### SUPPORT VECTOR MACHINES##########

## TRAIN THE MODELS 
svmfit_linear <- svm(y ~., data = bank_traindata, kernel = "linear", cost = 1, scale = TRUE)
svmfit_radial <- svm(y ~., data = bank_traindata, kernel = "radial", cost = 1, scale = TRUE, gamma =1)
svmfit_polynomial <- svm(y ~., data = bank_traindata, kernel = "polynomial", cost = 1, scale = TRUE, gamma =1)
svmfit_sigmoid <- svm(y ~., data = bank_traindata, kernel = "sigmoid", cost = 1, scale = TRUE, gamma =1)

#tuned <- tune(svm, y ~., data = bank_traindata, kernel = "linear", ranges = list(cost=c(0.01,0.1,1,10)))
# Will show the optimal cost parameter
#summary(tuned)

print(svmfit_linear)
print(svmfit_radial)
print(svmfit_polynomial)
print(svmfit_sigmoid)

#plot(svmfit, bank_traindata[,col])
#plot(svmfit_linear,data=bank_traindata, duration ~ balance)
#plot(svmfit_radial,data=bank_traindata, duration ~ balance)
#plot(svmfit_polynomial,data=bank_traindata, duration ~ balance)
#plot(svmfit_sigmoid,data=bank_traindata, duration ~ balance)

# Error Rates on Training Data
pt_linear = predict(svmfit_linear, bank_traindata[,-17], type ="class")
pt_radial = predict(svmfit_radial, bank_traindata[,-17], type ="class")
pt_polynomial = predict(svmfit_polynomial, bank_traindata[,-17], type ="class")
pt_sigmoid = predict(svmfit_sigmoid, bank_traindata[,-17], type ="class")

# Confusion Matrix - Training Data				
confusionMatrix(bank_traindata[,17], pt_linear)
confusionMatrix(bank_traindata[,17], pt_radial)
confusionMatrix(bank_traindata[,17], pt_polynomial)
confusionMatrix(bank_traindata[,17], pt_sigmoid)

# Error Rates on Test Data
p_bank_linear = predict(svmfit_linear, bank_testdata[,-17], type ="class")
p_bank_radial = predict(svmfit_radial, bank_testdata[,-17], type ="class")
p_bank_polynomial = predict(svmfit_polynomial, bank_testdata[,-17], type ="class")
p_bank_sigmoid = predict(svmfit_sigmoid, bank_testdata[,-17], type ="class")


plot(p_bank_linear)
table(p_bank_linear, bank_testdata[,17])

# Confusion Matrix - Test Data
confusionMatrix(bank_testdata[,17], p_bank_linear)
confusionMatrix(bank_testdata[,17], p_bank_radial)
confusionMatrix(bank_testdata[,17], p_bank_polynomial)
confusionMatrix(bank_testdata[,17], p_bank_sigmoid)


#### Varying the data-size vs./ Training and Testing Errors

# Define Data Indices for varying Data - Size
dataind = c(10000,15000,20000,25000,30000,31647)
Tr_err_SVM = c(0,0,0,0,0,0)
Tst_err_SVM = c(0,0,0,0,0,0)
# cf_min = 0.2


# Looping to calculate Training and Testing Errors for different Training Datasizes
for(i in 1:length(dataind))
{
  train_subset = bank_traindata[1:dataind[i],]
  
  # Fitting Training data using SVM, Radial Kernel
  svmfit_rad1 <- svm(y ~., data = train_subset, kernel = "radial", cost = 1, scale = TRUE, gamma=1)

  
  # Training and test prediction
  train = predict(svmfit_rad1, train_subset[,-17], type ="class")
  predicted = predict(svmfit_rad1, bank_testdata[,-17], type= "class")
  
  # Training and testing errors
  Tr_err_SVM[i] = 1- sum( train == train_subset$y ) / length( train )
  Tst_err_SVM[i] = 1- sum(predicted == bank_testdata$y ) / length( predicted )
  
}

# Create Data Frame to store Training and testing errors
subdata_svm_df = data.frame(dataind,Tr_err_SVM,Tst_err_SVM)  

# Plot Training and Test errors as a function of data-size
print(ggplot(subdata_svm_df, aes(dataind)) +                    
        geom_line(aes(y=Tr_err_SVM), colour="red", size = 2) +  
        geom_line(aes(y=Tst_err_SVM), colour="blue", size = 2) + 
        labs(title = "SVM (Kernel:Radial) Error Curves [Train (Red) & Test (Blue)]", 
             y = "Train/Test_Error", x = "Data Size") +
        scale_color_discrete(name = "Legend", labels = c("Train Error", "Test Error")))
