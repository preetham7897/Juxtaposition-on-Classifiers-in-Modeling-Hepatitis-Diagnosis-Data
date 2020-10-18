#Remove Global Variables

rm(list=ls(all=TRUE))

#import library files

library('e1071')
library('rpart')
library('caret')
library('randomForest')

#data normalisation

data = read.csv('/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/Imputed Data.csv', header=TRUE)

#Normalising Data - Z-Score

data$Age = scale(data$Age)
data$Bilirubin = scale(data$Bilirubin)
data$Alk.Phosphate = scale(data$Alk.Phosphate)
data$Sgot = scale(data$Sgot)
data$Albumin = scale(data$Albumin)
data$Protime = scale(data$Protime)

#MCC Function

mcc = function(cm)
{
  mc = (sum(diag(cm))-(cm[3]+cm[2]))/sqrt(unname(apply(cm, 1, sum))[1]*unname(apply(cm, 1, sum))[2]*unname(apply(cm, 2, sum))[1]*unname(apply(cm, 2, sum))[2])
  return(mc)
}

#Initialising values for output dataframe

ac1_tr = ac1_ts = ac2_tr = ac2_ts = ac3_ts = ac3_tr = ac4_tr = ac4_ts = ac5_tr = ac5_ts = c()
bac1_tr = bac1_ts = bac2_tr = bac2_ts = bac3_ts = bac3_tr = bac4_tr = bac4_ts = bac5_tr = bac5_ts = c()
r1_tr = r1_ts = r2_tr = r2_ts = r3_tr = r3_ts = r4_tr = r4_ts = r5_tr = r5_ts = c()
sp1_tr = sp1_ts = sp2_tr = sp2_ts = sp3_tr = sp3_ts = sp4_tr = sp4_ts = sp5_tr = sp5_ts = c()
pr1_tr = pr1_ts = pr2_tr = pr2_ts = pr3_tr = pr3_ts = pr4_tr = pr4_ts = pr5_tr = pr5_ts = c()
np1_tr = np1_ts = np2_tr = np2_ts = np3_ts = np3_tr = np4_tr = np4_ts = np5_ts = np5_tr = c()
f1_tr = f1_ts = f2_tr = f2_ts = f3_tr = f3_ts = f4_tr = f4_ts = f5_tr = f5_ts = c()
mc1_tr = mc1_ts = mc2_tr = mc2_ts = mc3_tr = mc3_ts = mc4_tr = mc4_ts = mc5_tr = mc5_ts = c()
uc1_tr = uc1_ts = uc2_tr = uc2_ts = uc3_tr = uc3_ts = uc4_tr = uc4_ts = uc5_tr = uc5_ts = c()

#finding averaged performance measures

for(i in 1:10)
{
  
  #Train Test Splitting
  
  index = createDataPartition(data$Class, p=0.8, list=F)
  train = data[index,]
  test = data[-index,]
  
  #Prediction with SVM
  
  m1 = svm(Class~., data=train)
  p1_tr = predict(m1, train)
  p1_ts = predict(m1, test)
  cm1_tr = table(Predicted=p1_tr, Actual=train$Class)
  cm1_ts = table(Predicted=p1_ts, Actual=test$Class)
  
  #Prediction with Naive Bayes
  
  m2 = naiveBayes(Class~., data=train)
  p2_tr = predict(m2, train)
  p2_ts = predict(m2, test)
  cm2_tr = table(Predicted=p2_tr, Actual=train$Class)
  cm2_ts = table(Predicted=p2_ts, Actual=test$Class)
  
  #Prediction with Decision Tree
  
  m3 = rpart(Class~., data=train)
  p3_tr = predict(m3, train, type="class")
  p3_ts = predict(m3, test, type="class")
  cm3_tr = table(Predicted=p3_tr, Actual=train$Class)
  cm3_ts = table(Predicted=p3_ts, Actual=test$Class)
  
  #Prediction with Random Forest
  
  m4 = randomForest(Class~., data=train)
  p4_tr = predict(m4, train)
  p4_ts = predict(m4, test)
  cm4_tr = table(Predicted=p4_tr, Actual=train$Class)
  cm4_ts = table(Predicted=p4_ts, Actual=test$Class)
  
  #Prediction with Logistic Regression
  
  glm_tr = train
  glm_ts = test
  glm_tr$Class = ifelse(glm_tr$Class == 'Die', 0, 1)
  glm_ts$Class = ifelse(glm_ts$Class == 'Die', 0, 1)
  m5 = glm(Class~., data=glm_tr, family=binomial(link='logit'))
  p5_tr = predict(m5, glm_tr, type='response')
  p5_ts = predict(m5, glm_ts, type='response')
  p5_tr = ifelse(p5_tr > 0.5, 1, 0)
  p5_ts = ifelse(p5_ts > 0.5, 1, 0)
  cm5_tr = table(Predicted=p5_tr, Actual=glm_tr$Class)
  cm5_ts = table(Predicted=p5_ts, Actual=glm_ts$Class)
  
  #Accuracy - Train
  
  ac1_tr = c(ac1_tr, sum(diag(cm1_tr))/sum(cm1_tr))
  ac2_tr = c(ac2_tr, sum(diag(cm2_tr))/sum(cm2_tr))
  ac3_tr = c(ac3_tr, sum(diag(cm3_tr))/sum(cm3_tr))
  ac4_tr = c(ac4_tr, sum(diag(cm4_tr))/sum(cm4_tr))
  ac5_tr = c(ac5_tr, sum(diag(cm5_tr))/sum(cm5_tr))
  
  #Accuracy - Test
  
  ac1_ts = c(ac1_ts, sum(diag(cm1_ts))/sum(cm1_ts))
  ac2_ts = c(ac2_ts, sum(diag(cm2_ts))/sum(cm2_ts))
  ac3_ts = c(ac3_ts, sum(diag(cm3_ts))/sum(cm3_ts))
  ac4_ts = c(ac4_ts, sum(diag(cm4_ts))/sum(cm4_ts))
  ac5_ts = c(ac5_ts, sum(diag(cm5_ts))/sum(cm5_ts))
  
  #Recall - Train

  r1 = unname(diag(cm1_tr)/apply(cm1_tr, 1, sum))[1]
  r1_tr = c(r1_tr, r1)
  
  r2 = unname(diag(cm2_tr)/apply(cm2_tr, 1, sum))[1]
  r2_tr = c(r2_tr, r2)
  
  r3 = unname(diag(cm3_tr)/apply(cm3_tr, 1, sum))[1]
  r3_tr = c(r3_tr, r3)
  
  r4 = unname(diag(cm4_tr)/apply(cm4_tr, 1, sum))[1]
  r4_tr = c(r4_tr, r4)
  
  r5 = unname(diag(cm5_tr)/apply(cm5_tr, 1, sum))[1]
  r5_tr = c(r5_tr, r5)
  
  #Recall - Test
  
  r6 = unname(diag(cm1_ts)/apply(cm1_ts, 1, sum))[1]
  r1_ts = c(r1_ts, r6)
  
  r7 = unname(diag(cm2_ts)/apply(cm2_ts, 1, sum))[1]
  r2_ts = c(r2_ts, r7)
  
  r8 = unname(diag(cm3_ts)/apply(cm3_ts, 1, sum))[1]
  r3_ts = c(r3_ts, r8)
  
  r9 = unname(diag(cm4_ts)/apply(cm4_ts, 1, sum))[1]
  r4_ts = c(r4_ts, r9)
  
  r0 = unname(diag(cm5_ts)/apply(cm5_ts, 1, sum))[1]
  r5_ts = c(r5_ts, r0)
  
  #Specificity - Train
  
  sp1 = unname(diag(cm1_tr)/apply(cm1_tr, 1, sum))[2]
  sp1_tr = c(sp1_tr, sp1)
  
  sp2 = unname(diag(cm2_tr)/apply(cm2_tr, 1, sum))[2]
  sp2_tr = c(sp2_tr, sp2)
  
  sp3 = unname(diag(cm3_tr)/apply(cm3_tr, 1, sum))[2]
  sp3_tr = c(sp3_tr, sp3)
  
  sp4 = unname(diag(cm4_tr)/apply(cm4_tr, 1, sum))[2]
  sp4_tr = c(sp4_tr, sp4)
  
  sp5 = unname(diag(cm5_tr)/apply(cm5_tr, 1, sum))[2]
  sp5_tr = c(r5_tr, sp5)
  
  #Specificity - Test
  
  sp6 = unname(diag(cm1_ts)/apply(cm1_ts, 1, sum))[2]
  sp1_ts = c(sp1_ts, sp6)
  
  sp7 = unname(diag(cm2_ts)/apply(cm2_ts, 1, sum))[2]
  sp2_ts = c(sp2_ts, sp7)
  
  sp8 = unname(diag(cm3_ts)/apply(cm3_ts, 1, sum))[2]
  sp3_ts = c(sp3_ts, sp8)
  
  sp9 = unname(diag(cm4_ts)/apply(cm4_ts, 1, sum))[2]
  sp4_ts = c(sp4_ts, sp9)
  
  sp0 = unname(diag(cm5_tr)/apply(cm5_tr, 1, sum))[2]
  sp5_ts = c(sp5_tr, sp0)
  
  #Precision - Train
  
  pr1 = unname(diag(cm1_tr)/apply(cm1_tr, 2, sum))[1]
  pr1_tr = c(pr1_tr, pr1)
  
  pr2 = unname(diag(cm2_tr)/apply(cm2_tr, 2, sum))[1]
  pr2_tr = c(pr2_tr, pr2)
  
  pr3 = unname(diag(cm3_tr)/apply(cm3_tr, 2, sum))[1]
  pr3_tr = c(pr3_tr, pr3)
  
  pr4 = unname(diag(cm4_tr)/apply(cm4_tr, 2, sum))[1]
  pr4_tr = c(pr4_tr, pr4)
  
  pr5 = unname(diag(cm5_tr)/apply(cm5_tr, 2, sum))[1]
  pr5_tr = c(pr5_tr, pr5)
  
  #Precision - Test
  
  pr6 = unname(diag(cm1_ts)/apply(cm1_ts, 2, sum))[1]
  pr1_ts = c(pr1_ts, pr6)
  
  pr7 = unname(diag(cm2_ts)/apply(cm2_ts, 2, sum))[1]
  pr2_ts = c(pr2_ts, pr7)
  
  pr8 = unname(diag(cm3_ts)/apply(cm3_ts, 2, sum))[1]
  pr3_ts = c(pr3_ts, pr8)
  
  pr9 = unname(diag(cm4_ts)/apply(cm4_ts, 2, sum))[1]
  pr4_ts = c(pr4_ts, pr9)
  
  pr0 = unname(diag(cm5_ts)/apply(cm5_ts, 2, sum))[1]
  pr5_ts = c(pr5_ts, pr0)
  
  #Balanced Accuracy - Train
  
  bac1_tr = c(bac1_tr, (r1+sp1)/2)
  bac2_tr = c(bac2_tr, (r2+sp2)/2)
  bac3_tr = c(bac3_tr, (r3+sp3)/2)
  bac4_tr = c(bac4_tr, (r4+sp4)/2)
  bac5_tr = c(bac5_tr, (r5+sp5)/2)
  
  #Balanced Accuracy - Test
  
  bac1_ts = c(bac1_ts, (r6+sp6)/2)
  bac2_ts = c(bac2_ts, (r7+sp7)/2)
  bac3_ts = c(bac3_ts, (r8+sp8)/2)
  bac4_ts = c(bac4_ts, (r9+sp9)/2)
  bac5_ts = c(bac5_ts, (r0+sp0)/2)
  
  #Negative Predicted Value - Train
  
  np1 = unname(diag(cm1_tr)/apply(cm1_tr, 2, sum))[2]
  np1_tr = c(np1_tr, np1)
  
  np2 = unname(diag(cm2_tr)/apply(cm2_tr, 2, sum))[2]
  np2_tr = c(np2_tr, np2)
  
  np3 = unname(diag(cm3_tr)/apply(cm3_tr, 2, sum))[2]
  np3_tr = c(np3_tr, np3)
  
  np4 = unname(diag(cm4_tr)/apply(cm4_tr, 2, sum))[2]
  np4_tr = c(np4_tr, np4)
  
  np5 = unname(diag(cm5_tr)/apply(cm5_tr, 2, sum))[2]
  np5_tr = c(np5_tr, np5)
  
  #Negative Predicted Value - Test
  
  np6 = unname(diag(cm1_ts)/apply(cm1_ts, 2, sum))[2]
  np1_ts = c(np1_ts, np6)
  
  np7 = unname(diag(cm2_ts)/apply(cm2_ts, 2, sum))[2]
  np2_ts = c(np2_ts, np7)
  
  np8 = unname(diag(cm3_ts)/apply(cm3_ts, 2, sum))[2]
  np3_ts = c(np3_ts, np8)
  
  np9 = unname(diag(cm4_ts)/apply(cm4_ts, 2, sum))[2]
  np4_ts = c(np4_ts, np9)
  
  np0 = unname(diag(cm5_ts)/apply(cm5_ts, 2, sum))[2]
  np5_ts = c(np5_ts, np0)
  
  #F1-measure - Train
  
  f1_tr = c(f1_tr, 2*pr1*r1/(pr1+r1))
  f2_tr = c(f2_tr, 2*pr2*r2/(pr2+r2))
  f3_tr = c(f3_tr, 2*pr3*r3/(pr3+r3))
  f4_tr = c(f4_tr, 2*pr4*r4/(pr4+r4))
  f5_tr = c(f5_tr, 2*pr5*r5/(pr5+r5))
  
  #F1-measure - Test
  
  f1_ts = c(f1_ts, 2*pr6*r6/(pr6+r6))
  f2_ts = c(f2_ts, 2*pr7*r7/(pr7+r7))
  f3_ts = c(f3_ts, 2*pr8*r8/(pr8+r8))
  f4_ts = c(f4_ts, 2*pr9*r9/(pr9+r9))
  f5_ts = c(f5_ts, 2*pr0*r0/(pr0+r0))
  
  #MCC - Train
  
  mc1_tr = c(mc1_tr, mcc(cm1_tr))
  mc2_tr = c(mc2_tr, mcc(cm2_tr))
  mc3_tr = c(mc3_tr, mcc(cm3_tr))
  mc4_tr = c(mc4_tr, mcc(cm4_tr))
  mc5_tr = c(mc5_tr, mcc(cm5_tr))
  
  #MCC - Test
  
  mc1_ts = c(mc1_ts, mcc(cm1_ts))
  mc2_ts = c(mc2_ts, mcc(cm2_ts))
  mc3_ts = c(mc3_ts, mcc(cm3_ts))
  mc4_ts = c(mc4_ts, mcc(cm4_ts))
  mc5_ts = c(mc5_ts, mcc(cm5_ts))
  
}

#Final Result Table

name = c('Accuracy', 'Balanced Accuracy', 'Recall', 'Specificity', 'Precision', 'Negative Predictive Value (NPV)', 'False Positive Rate (FPR)', 'False Discovery Rate (FDR)', 'False Negative Rate (FNR)', 'F-Measure', 'Mathews Correlation Coeffiecient (MCC)', 'Informedness', 'Markedness')
svm_tr = c(mean(ac1_tr), mean(bac1_tr), mean(r1_tr), mean(sp1_tr), mean(pr1_tr), mean(np1_tr), 1-mean(sp1_tr), 1-mean(pr1_tr), 1-mean(r1_tr), mean(f1_tr), mean(mc1_tr), mean(sp1_tr)+mean(r1_tr)-1, mean(pr1_tr)+mean(np1_tr)-1)
svm_ts = c(mean(ac1_ts), mean(bac1_ts), mean(r1_ts), mean(sp1_ts), mean(pr1_ts), mean(np1_ts), 1-mean(sp1_ts), 1-mean(pr1_ts), 1-mean(r1_ts), mean(f1_ts), mean(mc1_ts), mean(sp1_ts)+mean(r1_ts)-1, mean(pr1_ts)+mean(np1_ts)-1)
nb_tr = c(mean(ac2_tr), mean(bac2_tr), mean(r2_tr), mean(sp2_tr), mean(pr2_tr), mean(np2_tr), 1-mean(sp2_tr), 1-mean(pr2_tr), 1-mean(r2_tr), mean(f2_tr), mean(mc2_tr), mean(sp2_tr)+mean(r2_tr)-1, mean(pr2_tr)+mean(np2_tr)-1)
nb_ts = c(mean(ac2_ts), mean(bac2_ts), mean(r2_ts), mean(sp2_ts), mean(pr2_ts), mean(np2_ts), 1-mean(sp2_ts), 1-mean(pr2_ts), 1-mean(r2_ts), mean(f2_ts), mean(mc2_ts), mean(sp2_ts)+mean(r2_ts)-1, mean(pr2_ts)+mean(np2_ts)-1)
dt_tr = c(mean(ac3_tr), mean(bac3_tr), mean(r3_tr), mean(sp3_tr), mean(pr3_tr), mean(np3_tr), 1-mean(sp3_tr), 1-mean(pr3_tr), 1-mean(r3_tr), mean(f3_tr), mean(mc3_tr), mean(sp3_tr)+mean(r3_tr)-1, mean(pr3_tr)+mean(np3_tr)-1)
dt_ts = c(mean(ac3_ts), mean(bac3_ts), mean(r3_ts), mean(sp3_ts), mean(pr3_ts), mean(np3_ts), 1-mean(sp3_ts), 1-mean(pr3_ts), 1-mean(r3_ts), mean(f3_ts), mean(mc3_ts), mean(sp3_ts)+mean(r3_ts)-1, mean(pr3_ts)+mean(np3_ts)-1)
rf_tr = c(mean(ac4_tr), mean(bac4_tr), mean(r4_tr), mean(sp4_tr), mean(pr4_tr), mean(np4_tr), 1-mean(sp4_tr), 1-mean(pr4_tr), 1-mean(r4_tr), mean(f4_tr), mean(mc4_tr), mean(sp4_tr)+mean(r4_tr)-1, mean(pr4_tr)+mean(np4_tr)-1)
rf_ts = c(mean(ac4_ts), mean(bac4_ts), mean(r4_ts), mean(sp4_ts), mean(pr4_ts), mean(np4_ts), 1-mean(sp4_ts), 1-mean(pr4_ts), 1-mean(r4_ts), mean(f4_ts), mean(mc4_ts), mean(sp4_ts)+mean(r4_ts)-1, mean(pr4_ts)+mean(np4_ts)-1)
lr_tr = c(mean(ac5_tr), mean(bac5_tr), mean(r5_tr), mean(sp5_tr), mean(pr5_tr), mean(np5_tr), 1-mean(sp5_tr), 1-mean(pr5_tr), 1-mean(r5_tr), mean(f5_tr), mean(mc5_tr), mean(sp5_tr)+mean(r5_tr)-1, mean(pr5_tr)+mean(np5_tr)-1)
lr_ts = c(mean(ac5_ts), mean(bac5_ts), mean(r5_ts), mean(sp5_ts), mean(pr5_ts), mean(np5_ts), 1-mean(sp5_ts), 1-mean(pr5_ts), 1-mean(r5_ts), mean(f5_ts), mean(mc5_ts), mean(sp5_ts)+mean(r5_ts)-1, mean(pr5_ts)+mean(np5_ts)-1)

df_tr = data.frame(Parameters=name, 'SVM Train'=svm_tr, 'Naive Bayes Train'=nb_tr, 'Decision Tree Train'=dt_tr, 'Random Forest Train'=rf_tr, 'Logistic Regression Train'=lr_tr)
df_ts = data.frame(Parameters=name, 'SVM Test'=svm_ts, 'Naive Bayes Test'=nb_ts, 'Decision Tree Test'=dt_ts, 'Random Forest Test'=rf_ts, 'Logistic Regression Test'=lr_ts)

write.csv(df_tr, file="/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/Results/main_result_train.csv", row.names = FALSE)
write.csv(df_ts, file="/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/Results/main_result_test.csv", row.names = FALSE)

#Plots

n = c('NB', 'DT', 'LR', 'SVM', 'RF')
p1 = c(nb_ts[1], dt_ts[1], lr_ts[1], svm_ts[1], rf_ts[1])
png(file = ('/home/preetham/Documents/Research Projects/Hepatitis Classification/acc_comp.png'))
barplot(p1, names.arg=n, xlab="Model", ylab="Measure", col="blue", main="Comparison on Accuracy", border="red")
dev.off()
