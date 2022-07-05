library(ggplot2)
library(tidyverse)
library(dplyr)
library(caret)
library(mlbench)
library(tidyr)
library(Boruta)
library(data.table)
library(modelr)
library(cowplot)
library(randomForest)
library(ROSE)
library(e1071)
library(Matrix)
library(xgboost)
library(mlr)
library(MLeval)

# Reading the demographics, laboratory and examination .csv files using read_csv

demo <- read_csv("C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/demographic.csv")
labs <- read_csv("C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/labs.csv")
exam <- read_csv("C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/examination.csv")


# Making new dataframes consisting of only selected columns
demo_df <- demo %>% select(SEQN , "GENDER" = RIAGENDR, "AGE" = RIDAGEYR)

# HBA1c is Hemoglobin A1c or Glycohemoglobin which is an indicator of diabetes
lab_df <- labs %>% select(SEQN, "HbA1c" = LBXGH)

# Creating another dataframe by inner joining the other data frames
df <- lab_df %>% inner_join(demo_df, by = "SEQN") %>%
                            inner_join(exam, by = "SEQN")

summary(df)

# Keeping columns which have less than or equal to 50% missing values (NAs)

DF <- df[apply(df,2,function(x) mean(is.na(x))<=.50)] 

# We can see that there are no more columns which have more than 50% missing
#values

write_csv(DF, "C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/checking.csv")

# Discarding remaining irrelevant features from dataframe and selecting only 
#useful features 
# Gender not mentioned in documentation

DF1 <- DF %>% select(SEQN, HbA1c,  AGE, "BP_SYS" = BPXSY1,
                     "BP_DIA" = BPXDI1, "WEIGHT" = BMXWT, "HEIGHT" = BMXHT,
                     "BMI" = BMXBMI, "UPPER_LEG_LENGTH" = BMXLEG, 
                     "UPPER_ARM_LENGTH" = BMXARML, "ARM_CIRCUM" = BMXARMC, 
                     "WAIST_CIRCUMF" = BMXWAIST, "SAG_ABDOMINAL_DIA" = BMDAVSAD)


# 1) How does height vary with age? Are there any anomalies?
# Plot an age vs Height Graph
DF1 %>%
  filter(AGE<60) %>%
  ggplot(aes(x = as.factor(AGE), y = HEIGHT)) + 
  geom_boxplot(color = "blue", fill = "blue", 
               alpha = 0.2, notch = TRUE, notchwidth = 0.8, outlier.colour = "red",
               outlier.fill = "red", outlier.size = 3, outlier.alpha = 1) + 
  labs(title = "Variation of height with age",
       x= "Age",
       y = "Height in cm")
  

# Till about 18 years height increases with age, but after that it seems to 
#be constant which is as expected


# Removing observations less than 5 years old
DF1 <- subset(DF1, AGE>=5)

# Imputing missing values of remaining columns with median
f=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it 
                                  #is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value 
                                      #from the column
  x #display the column
}

Final_DF = data.frame(apply(DF1,2,f))
f_df <- data.frame(apply(DF1,2,f))

summary(Final_DF)

# Changing the HbA1c values to yes or no diabetes

#No Diabetes 6.4 and lower Yes Diabetes 6.5 or higher

Final_DF$HbA1c <- cut(Final_DF$HbA1c,
                      breaks = c(-Inf, 6.5, Inf),
                      labels = c("NO", "YES"),
                      right = FALSE) # Interval is closed on the left and open 
                                      #on right

names(Final_DF)[names(Final_DF) == "HbA1c"] <- "Diabetes" # Renaming column
str(Final_DF)




# 2) Number of people who have vs don't have diabetes
Final_DF %>%
  ggplot(aes(x = Diabetes)) + geom_bar(data = Final_DF, fill = "blue", 
                                       color = "black", alpha  = 0.7) +
  labs(title = "Number of people that have diabetes vs number of people that don't have diabetes")
###
#EDA
###


#SEQN, HbA1c,  AGE, "BP_SYS" = BPXSY1,
#"BP_DIA" = BPXDI1, "WEIGHT" = BMXWT, "HEIGHT" = BMXHT,
#"BMI" = BMXBMI, "UPPER_LEG_LENGTH" = BMXLEG, 
#"UPPER_ARM_LENGTH" = BMXARML, "ARM_CIRCUM" = BMXARMC, 
#"WAIST_CIRCUMF" = BMXWAIST, "SAG_ABDOMINAL_DIA" = BMDAVSAD


# Make a dataset with people with only diabetes

diabetes_data <- Final_DF %>%
  filter(Diabetes == "YES")

x1 <- diabetes_data %>%
  filter(AGE>5) %>%
  filter(AGE<=15) %>%
  count(Diabetes == "YES")

x2 <- diabetes_data %>%
  filter(AGE>15) %>%
  filter(AGE<=25) %>%
  count(Diabetes == "YES")

x3 <- diabetes_data %>%
  filter(AGE>25) %>%
  filter(AGE<=35) %>%
  count(Diabetes == "YES")

x4 <- diabetes_data %>%
  filter(AGE>35) %>%
  filter(AGE<=45) %>%
  count(Diabetes == "YES")

x5 <- diabetes_data %>%
  filter(AGE>45) %>%
  filter(AGE<=55) %>%
  count(Diabetes == "YES")

x6 <- diabetes_data %>%
  filter(AGE>55) %>%
  filter(AGE<=65) %>%
  count(Diabetes == "YES")

x7 <- diabetes_data %>%
  filter(AGE>65) %>%
  count(Diabetes == "YES")

age_groups <- x1 %>% full_join(x2)%>% full_join(x3)%>% 
  full_join(x4)%>% full_join(x5)%>% full_join(x6)%>% full_join(x7)

col_names <- c("16 to 25", "26 to 35", "36 to 45", "46 to 55", "56 to 65",
               "66 to 80" )
age_g <- c( 7, 13, 71, 124, 155, 244)

x8 <- data.frame(col_names, age_g)
x8 <- x8 %>%
  group_by(age_g)


# 3) What age group seems to have diabetes more frequently

ggplot(x8, aes(x=col_names,y = age_g, fill = age_g)) +  geom_col() + 
  scale_fill_viridis_c(alpha = 0.95) +
  geom_text(aes(label = age_g), vjust = 0.01, size = 5) + 
  labs(title = "Cases of diabetes in different age groups",
       x = "Age groups",
       y = "Number of cases")

# 4) Distribution of people having diabetes with AGE

ggplot(Final_DF, aes(x = Diabetes, y = AGE)) + 
  geom_boxplot(color = "blue", fill = "blue", alpha = 0.2, 
               notch = TRUE, notchwidth = 0.8, outlier.colour = "red",
                                                            outlier.fill = "red",
               outlier.size = 3, outlier.alpha = 1) + 
  labs(title = "Distribution of people having diabetes with age",
       x = "Diabetes",
       y = "Age")


# 5) Relation between BMI and diabetes

 
bmi_under <- diabetes_data %>%
  filter(BMI<18.5) %>%
  count(Diabetes == "YES")

bmi_normal <- diabetes_data %>%
  filter(BMI>=18.5) %>%
  filter(BMI<25) %>%
  count(Diabetes == "YES")

bmi_over <- diabetes_data %>%
  filter(BMI>=25) %>%
  filter(BMI<30) %>%
  count(Diabetes == "YES")

bmi_obese <- diabetes_data %>%
  filter(BMI>=30) %>%
  count(Diabetes == "YES")

bmi_groups <- bmi_under %>% full_join(bmi_normal)%>% full_join(bmi_over)%>%
  full_join(bmi_obese)

bx <- c("Underweight", "Normal", "Overweight", "Obese")
by <- c( 3, 67,187, 347)

bd <- data.frame(bx, by)

ggplot(bd, aes(x=bx,y = by, fill = bx)) +  geom_col() + 
  scale_fill_viridis_d(alpha = 0.95) +
  geom_text(aes(label = by), vjust = 0.01, size = 5) + 
  labs(title = "Cases of diabetes in different weight groups",
       x = "Weight groups",
       y = "Number of cases")

#Correlation Plot
library(ggcorrplot)

corr <- round(cor(f_df), 1)

ggcorrplot(corr, hc.order = TRUE,
           lab = TRUE,
           lab_size = 3,
           method = "circle",
           colors = c("tomato2", "white", "springgreen3"),
           title = "Correlation plot of dataset",
           ggtheme = theme_bw())

## Feature importance using Boruta Package

boruta_output <- Boruta(Diabetes ~ . , data=na.omit(Final_DF), maxRuns = 200,
                        doTrace=2)
print(boruta_output)
#names(boruta_output)

boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)

roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Variable Importance Scores
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
imps2[order(-imps2$meanImp), ]  # descending sort
setDT(imps2, keep.rownames = TRUE)[]

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance") 

## Remove SEQN from df

Final_DF_1 <- subset(Final_DF , select  = -c(SEQN))

#####################################
#Fitting Models
####################################


#################
##Splitting Data
#################

dat_part <- resample_partition(Final_DF_1,
                               c(train = 0.8,
                               test = 0.2))
                            

train <- as.tibble(dat_part$train)
test <- as.tibble(dat_part$test)

##############
#Oversampling
##############

train.rose <- ovun.sample(Diabetes~. , data = train, method = "over", 
                          seed = 1)$data

table(train.rose$Diabetes)
table(train$Diabetes)
summary(train)
summary(train.rose)

##############
#Undersampling
##############


train.rose_under <- ovun.sample(Diabetes~. , data = train, method = "under",
                                N = 964, seed = 1)$data

table(train.rose_under$Diabetes)
table(train$Diabetes)
summary(train)
summary(train.rose_under)


#########################
#Logistic Regression
#########################

##################
# Normal Sample
#################

log_train <- train
log_test <- test

fit_log_normal <- glm(Diabetes ~ . , family = binomial(link = "logit"),
                      data = log_train)

summary(fit_log_normal)


log_test <- 
  log_test %>%
  add_predictions(fit_log_normal, type = "response") %>%
  mutate(pred_dia = ifelse(pred > 0.12,
                              "YES", "NO"),
         correct = Diabetes == pred_dia )


table(log_test$Diabetes, log_test$pred_dia)[,1:2] # Confusion Matrix

# Sensitivity = 0.6885246

# Specificity = 0.8261421

# Accuracy = 0.8162544

##################
# Oversampling
#################

log_train_over <- train.rose
log_test_over <- test

fit_log_over <- glm(Diabetes ~ . , family = binomial(link = "logit"),
                    data = log_train_over)
summary(fit_log_over)

log_test_over <- 
  log_test_over %>%
  add_predictions(fit_log_over, type = "response") %>%
  mutate(pred_dia = ifelse(pred > 0.5,
                           "YES", "NO"),
         correct = Diabetes == pred_dia )


table(log_test_over$Diabetes, log_test_over$pred_dia)[,1:2] # Confusion Matrix

#sensitivity = 0.852459

#specificity = 0.732868

#accuracy = 0.0.7414065

##################
# Undersampling
#################

log_train_under <- train.rose_under
log_test_under <- test

fit_log_under <- glm(Diabetes ~ . , family = binomial(link = "logit"),
                     data = log_train_under)
summary(fit_log_under)

log_test_under <- 
  log_test_under %>%
  add_predictions(fit_log_under, type = "response") %>%
  mutate(pred_dia = ifelse(pred > 0.5,
                           "YES", "NO"),
         correct = Diabetes == pred_dia )


table(log_test_under$Diabetes, log_test_under$pred_dia)[,1:2] # Confusion Matrix

#sensitivity = 0.8852459

#specificity = 0.7277919

#accuracy = 0.7391048



######################
##Random Forests
######################

## Normal Sample


summary(train)


model_normal_rf <- randomForest(Diabetes ~ ., data = train, ntree = 1000, 
                                proximity = TRUE) # Prox = true returns 
                                                            #proximity matrix

model_normal_rf

prediction <- as.data.frame((predict(model_normal_rf, newdata = test)))

confusionMatrix((predict(model_normal_rf, newdata = test)), test$Diabetes,
                positive = "YES") # Bad Sensitivity but good specificity

#sensitivity = 0.049180 

#specificity = 0.995558 

#accuracy = 0.9276 

# save the model to disk
saveRDS(model_normal_rf, "C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/normalsample_rf.rds")

## Oversampling


model <- randomForest(Diabetes ~ ., data = train.rose, ntree = 1000,
                      proximity = TRUE) # Prox = true returns proximity matrix

model


confusionMatrix((predict(model, newdata = test)), test$Diabetes,
                positive = "YES") # Bad Sensitivity but good specificity

#sensitivity = 0.07377

#specificity = 0.98731

#accuracy = 0.9217

oob.error.data <- data.frame(
  Trees = rep(1:nrow(model$err.rate), times=3),
  Type = rep(c("OOB", "NO", "YES"), each = nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"],
          model$err.rate[,"NO"],
          model$err.rate[,"YES"]))

ggplot(data = oob.error.data, aes(x=Trees, y=Error)) + 
  geom_line(aes(color = Type))

oob.values <- vector(length = 12)
for(i in 1:12){
  temp.model <- randomForest(Diabetes~., data = train.rose, mtry=i,
                             ntree = 1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}

oob.values # mtry = 1 gave the best oob

# save the model to disk
saveRDS(model, "C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/oversampled_rf.rds")


#code to load model
#super_model <- readRDS("")
#print(super_model)

## Undersampling


model_under <- randomForest(Diabetes ~ ., data = train.rose_under, ntree = 1000,
                            proximity = TRUE) # Prox = true returns proximity
                                                                        #matrix

model_under


confusionMatrix(predict(model_under, test), test$Diabetes, positive = "YES")
# Good sensitivity

#sensitivity = Sensitivity : 0.90164

#specificity = 0.71447

#accuracy = 0.7279

oob.error.data <- data.frame(
  Trees = rep(1:nrow(model$err.rate), times=3),
  Type = rep(c("OOB", "NO", "YES"), each = nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"],
          model$err.rate[,"NO"],
          model$err.rate[,"YES"]))

ggplot(data = oob.error.data, aes(x=Trees, y=Error)) + 
  geom_line(aes(color = Type))

oob.values <- vector(length = 12)
for(i in 1:12){
  temp.model <- randomForest(Diabetes~., data = train.rose, mtry=i, 
                             ntree = 1000)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}  ##ntrees = 630 seems to be the best

oob.values # mtry = 1 gave the best oob


saveRDS(model_under, "C:/Semester 2/Intro to Data Mining and Processing/Project/national-health-and-nutrition-examination-survey/undersampled_rf.rds")


#tuning the model

model_under_tuned <- randomForest(Diabetes ~ ., data = train.rose_under,
                                  ntree = 1000, proximity = TRUE)
# Prox = true returns proximity matrix
model_under_tuned

confusionMatrix(predict(model_under_tuned, test), test$Diabetes,
                positive = "YES")

          
#Sensitivity : 0.90984         
#Specificity : 0.71256  
#Accuracy : 0.7267


###################
##xgboost
##################

###################
##NORMAL DATA
###################

df_train_0 <- train.rose
df_test_0 <- test

setDT(df_train_0)
setDT(df_test_0)

# Using one hot encoding

labels_0 <- df_train_0$Diabetes
ts_label_0 <- df_test_0$Diabetes

new_tr_0 <- model.matrix(~.+0, data = df_train_0[,-c("Diabetes"), with=F])

new_ts_0 <- model.matrix(~.+0, data = df_test_0[,-c("Diabetes"), with=F])

#convert factor to numeric 
labels_0 <- as.numeric(labels_0)-1
ts_label_0 <- as.numeric(ts_label_0)-1

dtrain_0 <- xgb.DMatrix(data = new_tr_0,label = labels_0) 
dtest_0 <- xgb.DMatrix(data = new_ts_0,label=ts_label_0)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, 
               gamma=0, max_depth=6, min_child_weight=1, subsample=1, 
               colsample_bytree=1)

xgbcv_0 <- xgb.cv( params = params, data = dtrain_0, nrounds = 100, nfold = 5,
                   showsd = T, stratified = T, print.every.n = 10, 
                   early.stop.round = 20, maximize = F) 
# lowest in 100th iteration

#first default - model training
xgb1_0 <- xgb.train (params = params, data = dtrain_0, nrounds = 79, watchlist =
                     list(val=dtest_0,train=dtrain_0), print.every.n = 10, 
                     early.stop.round = 10, maximize = F , 
                     eval_metric = "error")

#model prediction
xgbpred_0 <- predict (xgb1_0,dtest_0)
xgbpred_0 <- ifelse (xgbpred_0 > 0.5,'YES','NO')
xgbpred1_0 <- as.factor(xgbpred_0)
confusionMatrix(xgbpred1_0 , df_test_0$Diabetes, positive = "YES") 
# Bad sensitivity

#Sensitivity : 0.27049         
#Specificity : 0.93147 
#Accuracy : 0.884 

##################
##OVERSAMPLED DATA
##################
df_train <- train.rose
df_test <- test

setDT(df_train)
setDT(df_test)

# Using one hot encoding

labels <- df_train$Diabetes
ts_label <- df_test$Diabetes

new_tr <- model.matrix(~.+0, data = df_train[,-c("Diabetes"), with=F])

new_ts <- model.matrix(~.+0, data = df_test[,-c("Diabetes"), with=F])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3,
               gamma=0, max_depth=6, min_child_weight=1, subsample=1, 
               colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5,
                 showsd = T, stratified = T, print.every.n = 10,
                 early.stop.round = 20, maximize = F) 
# lowest in 100th iteration

#first default - model training
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 79, watchlist =
                     list(val=dtest,train=dtrain), print.every.n = 10, 
                   early.stop.round = 10, maximize = F , eval_metric = "error")

#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,'YES','NO')
xgbpred1 <- as.factor(xgbpred)
confusionMatrix(xgbpred1 , df_test$Diabetes, positive = "YES")
# very low sensitivity but high specificity

#Sensitivity : 0.27049         
#Specificity : 0.93147
#Accuracy : 0.884

###################
##UNDERSAMPLED DATA
###################

df_train1 <- train.rose_under
df_test1 <- test

setDT(df_train1)
setDT(df_test1)

# Using one hot encoding

labels1 <- df_train1$Diabetes
ts_label1 <- df_test1$Diabetes

new_tr1 <- model.matrix(~.+0, data = df_train1[,-c("Diabetes"), with=F])

new_ts1 <- model.matrix(~.+0, data = df_test1[,-c("Diabetes"), with=F])
#convert factor to numeric 
labels1 <- as.numeric(labels1)-1
ts_label1 <- as.numeric(ts_label1)-1

dtrain1 <- xgb.DMatrix(data = new_tr1,label = labels1) 
dtest1 <- xgb.DMatrix(data = new_ts1,label=ts_label1)

#default parameters

xgbcv1 <- xgb.cv( params = params, data = dtrain1, nrounds = 100, 
                  nfold = 5, showsd = T, stratified = T, print.every.n = 10, 
                  early.stop.round = 20, maximize = F) 
# lowest in 6th iteration

#first default - model training
xgb1_1 <- xgb.train (params = params, data = dtrain1, nrounds = 79, watchlist =
                     list(val=dtest1,train=dtrain1), print.every.n = 10,
                     early.stop.round = 10, maximize = F ,
                     eval_metric = "error")

#model prediction
xgbpred11 <- predict (xgb1_1,dtest1)
xgbpred11 <- ifelse (xgbpred11 > 0.5,'YES','NO')
xgbpred12 <- as.factor(xgbpred11)
confusionMatrix(xgbpred12 , df_test$Diabetes, positive = "YES")

#Sensitivity : 0.86885
#Specificity : 0.72145
#Accuracy : 0.732 



###################
##SVM
##################

##################
#normal sample
#################

svm_model_normal <- svm(Diabetes ~ ., data=train)
summary(svm_model)

confusionMatrix(predict (svm_model_normal, test) , df_test$Diabetes,
                positive = "YES")

# Sensitivity : 0.00000       
# Specificity : 1.00000 
# Accuracy : 0.9282

##################
#oversampled
#################

svm_model_over <- svm(Diabetes ~ ., data=train.rose)
summary(svm_model_over)

confusionMatrix(predict (svm_model_over, test) , df_test$Diabetes,
                positive = "YES")

#Sensitivity : 0.88525
#Specificity : 0.71510 
#Accuracy : 0.7273


##################
#undersampled
#################

svm_model_under <- svm(Diabetes ~ ., data=train.rose_under)
summary(svm_model)

confusionMatrix(predict (svm_model_under, test) , df_test$Diabetes,
                positive = "YES")

# Sensitivity : 0.92623
# Specificity : 0.67893
# Accuracy : 0.6967

########################
#Comparing Results
########################

