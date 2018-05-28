library(dummies)
library(rpart)
library(e1071)
library(caret)
library(rpart.plot)
library(rattle)
library(ggplot2)
library(C50)
library(xgboost)

set.seed(100)

# Import Dataset
creditdata = read.csv(file = "../credit-data.csv", header = TRUE, sep = ',')

# Create Training, Testing Data (70/30)
creditdatasample <- sample.int(n = nrow(creditdata), size = floor(.70*nrow(creditdata)), replace = F)
credit_traindata <- creditdata[creditdatasample, ]
credit_testdata  <- creditdata[-creditdatasample, ]

##### SVM ###########

# Fit SVMs with different kernels on Training Data
svm_credit_linear <- svm(A16 ~., data = credit_traindata, kernel = "linear", cost = 1, scale = TRUE)
svm_credit_radial <- svm(A16 ~., data = credit_traindata, kernel = "radial", cost = 1, scale = TRUE)
svm_credit_polynomial <- svm(A16 ~., data = credit_traindata, kernel = "polynomial", cost = 1, scale = TRUE)
svm_credit_sigmoid <- svm(A16 ~., data = credit_traindata, kernel = "sigmoid", cost = 1, scale = TRUE)

# Error Rates on Training Data
pt_credit_linear = predict(svm_credit_linear, credit_traindata[,-16], type ="class")
pt_credit_radial = predict(svm_credit_radial, credit_traindata[,-16], type ="class")
p_credit_polynomial = predict(svm_credit_polynomial, credit_testdata[,-16], type ="class")
pt_credit_sigmoid = predict(svm_credit_sigmoid, credit_traindata[,-16], type ="class")

# Confusion Matrix - Training Data				
confusionMatrix(credit_traindata[,16], pt_credit_linear)
confusionMatrix(credit_traindata[,16], pt_credit_radial)
confusionMatrix(credit_testdata[,16], p_credit_polynomial)
confusionMatrix(credit_traindata[,16], pt_credit_sigmoid)


# Error Rates on Test Data
p_credit_linear = predict(svm_credit_linear, credit_testdata[,-16], type ="class")
p_credit_radial = predict(svm_credit_radial, credit_testdata[,-16], type ="class")
p_credit_polynomial = predict(svm_credit_polynomial, credit_testdata[,-16], type ="class")
p_credit_sigmoid = predict(svm_credit_sigmoid, credit_testdata[,-16], type ="class")

# Confusion Matrix - Test Data				
confusionMatrix(credit_testdata[,16], p_credit_linear)
confusionMatrix(credit_testdata[,16], p_credit_radial)
confusionMatrix(credit_testdata[,16], p_credit_polynomial)
confusionMatrix(credit_testdata[,16], p_credit_sigmoid)

#### Varying the data-size vs./ Training and Testing Errors
dIndx = c(50,100,150,200,250,300,350,400,482)
Train_err_SVM = c(0,0,0,0,0,0,0,0,0)
Test_err_SVM = c(0,0,0,0,0,0,0,0,0)



# Looping to calculate Training and Testing Errors for different Training Datasizes
for(i in 1:length(dIndx))
{
  train_subset = credit_traindata[1:dIndx[i],]
  
  # Model fitting on Training Data - SVM Linear
  svmfit_lnr <- svm(A16 ~., data = train_subset, kernel = "linear", cost = 1, scale = TRUE)
  
  # Training and test prediction
  train = predict(svmfit_lnr, train_subset[,-16], type ="class")
  predicted = predict(svmfit_lnr, credit_testdata[,-16], type= "class")
  
  # Training and testing errors
  Train_err_SVM[i] = 1- sum(train == train_subset$A16 ) / length( train )
  Test_err_SVM[i] = 1- sum(predicted == credit_testdata$A16 ) / length( predicted )
  
}

# Create Data Frame to store Training and testing errors
credata_svm_df = data.frame(dIndx,Train_err_SVM,Test_err_SVM)  

# Plot Training and Test errors as a function of data-size
print(ggplot(credata_svm_df, aes(dIndx)) +                    
        geom_line(aes(y=Train_err_SVM), colour="red", size = 2) +  
        geom_line(aes(y=Test_err_SVM), colour="blue", size = 2) + 
        labs(title = "SVM (Kernel:Linear) Error Curves [Train (Red) & Test (Blue)]", 
             y = "Train/Test_Error", x = "Data Size") +
        scale_color_discrete(name = "Legend", labels = c("Train Error", "Test Error")))