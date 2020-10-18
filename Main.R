#import library files

library('e1071')
library('rpart')
library('caret')
library('ROCR')
library('mlbench')

#data normalisation

data = read.csv('/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/Imputed Data.csv', header=TRUE)

#Normalising Data - Z-Score

data$Age = scale(data$Age)
data$Bilirubin = scale(data$Bilirubin)
data$Alk.Phosphate = scale(data$Alk.Phosphate)
data$Sgot = scale(data$Sgot)
data$Albumin = scale(data$Albumin)
data$Protime = scale(data$Protime)

#data train test split

index = createDataPartition(data$Class, p=0.8, list=F)
train = data[index,]
test = data[-index,]

#Prediction with SVM

m1 = svm(Class~., data=train)
p1_tr = predict(m1, train)
p1_ts = predict(m1, test)
cm1_tr = table(Actual=train$Class, Predicted=p1_tr)
cm1_ts = table(Actual=test$Class, Predicted=p1_ts)

#SVM - Training Error different accuracy measures

n1_tr = sum(cm1_tr)
nc1_tr = nrow(cm1_tr)
diag1_tr = diag(cm1_tr)
rs1_tr = apply(cm1_tr, 1, sum)
cs1_tr = apply(cm1_tr, 2, sum)
a1_tr = rs1_tr/n1_tr
b1_tr = cs1_tr/n1_tr
er1_tr = 1-sum(diag1_tr)/n1_tr
ac1_tr = 1-er1_tr
pr1_tr = diag1_tr/cs1_tr
r1_tr = diag1_tr/rs1_tr
f11_tr = 2*pr1_tr*r1_tr/(pr1_tr+r1_tr)

#SVM - Testing Error different accuracy measures

n1_ts = sum(cm1_ts)
nc1_ts = nrow(cm1_ts)
diag1_ts = diag(cm1_ts)
rs1_ts = apply(cm1_ts, 1, sum)
cs1_ts = apply(cm1_ts, 2, sum)
a1_ts = rs1_ts/n1_ts
b1_ts = cs1_ts/n1_ts
er1_ts = 1-sum(diag1_ts)/n1_ts
ac1_ts = 1-er1_ts
pr1_ts = diag1_ts/cs1_ts
r1_ts = diag1_ts/rs1_ts
f11_ts = 2*pr1_ts*r1_ts/(pr1_ts+r1_ts)

#Prediction with Naive Bayes

m2 = naiveBayes(Class~., data=train)
p2_tr = predict(m2, train)
p2_ts = predict(m2, test)
cm2_tr = table(Actual=train$Class, Predicted=p2_tr)
cm2_ts = table(Actual=test$Class, Predicted=p2_ts)

#Naive Bayes - Training Error different accuracy measures

n2_tr = sum(cm2_tr)
nc2_tr = nrow(cm2_tr)
diag2_tr = diag(cm2_tr)
rs2_tr = apply(cm2_tr, 1, sum)
cs2_tr = apply(cm2_tr, 2, sum)
a2_tr = rs2_tr/n2_tr
b2_tr = cs2_tr/n2_tr
er2_tr = 1-sum(diag2_tr)/n2_tr
ac2_tr = 1-er2_tr
pr2_tr = diag2_tr/cs2_tr
r2_tr = diag2_tr/rs2_tr
f12_tr = 2*pr2_tr*r2_tr/(pr2_tr+r2_tr)

#Naive Bayes - Testing Error different accuracy measures

n2_ts = sum(cm2_ts)
nc2_ts = nrow(cm2_ts)
diag2_ts = diag(cm2_ts)
rs2_ts = apply(cm2_ts, 1, sum)
cs2_ts = apply(cm2_ts, 2, sum)
a2_ts = rs2_ts/n2_ts
b2_ts = cs2_ts/n2_ts
er2_ts = 1-sum(diag2_ts)/n2_ts
ac2_ts = 1-er2_ts
pr2_ts = diag2_ts/cs2_ts
r2_ts = diag2_ts/rs2_ts
f12_ts = 2*pr2_ts*r2_ts/(pr2_ts+r2_ts)

#Prediction with Decision Tree

m3 = rpart(Class~., data=train)
p3_tr = predict(m3, train, type="class")
p3_ts = predict(m3, test, type="class")
cm3_tr = table(Actual=train$Class, Predicted=p3_tr)
cm3_ts = table(Actual=test$Class, Predicted=p3_ts)

#Decision Tree - Training Error, different accuracy measures

n3_tr = sum(cm3_tr)
nc3_tr = nrow(cm3_tr)
diag3_tr = diag(cm3_tr)
rs3_tr = apply(cm3_tr, 1, sum)
cs3_tr = apply(cm3_tr, 2, sum)
a3_tr = rs3_tr/n3_tr
b3_tr = cs3_tr/n3_tr
er3_tr = 1-sum(diag3_tr)/n3_tr
ac3_tr = 1-er3_tr
pr3_tr = diag3_tr/cs3_tr
r3_tr = diag3_tr/rs3_tr
f13_tr = 2*pr3_tr*r3_tr/(pr3_tr+r3_tr)

#Decision Tree - Testing Error different accuracy measures

n3_ts = sum(cm3_ts)
nc3_ts = nrow(cm3_ts)
diag3_ts = diag(cm3_ts)
rs3_ts = apply(cm3_ts, 1, sum)
cs3_ts = apply(cm3_ts, 2, sum)
a3_ts = rs3_ts/n3_ts
b3_ts = cs3_ts/n3_ts
er3_ts = 1-sum(diag3_ts)/n3_ts
ac3_ts = 1-er3_ts
pr3_ts = diag3_ts/cs3_ts
r3_ts = diag3_ts/rs3_ts
f13_ts = 2*pr3_ts*r3_ts/(pr3_ts+r3_ts)

#Prediction with KNN

x = trainControl(method="repeatedcv", number=3, repeats=3, classProbs=TRUE, summaryFunction=twoClassSummary)
m4 = train(Class~., data=train, method="knn", preProcess=c("center","scale"),trControl=x, metric="ROC", tuneLength=12)
p4_tr = predict(m4, train, type="prob")
p4_ts = predict(m4, test, type="prob")
cm4_tr = table(Actual=train$Class, Predicted=p4_tr)
cm4_ts = table(Actual=test$Class, Predicted=p4_ts)

#KNN - Training Error, different accuracy measures

n4_tr = sum(cm4_tr)
nc4_tr = nrow(cm4_tr)
diag4_tr = diag(cm4_tr)
rs4_tr = apply(cm4_tr, 1, sum)
cs4_tr = apply(cm4_tr, 2, sum)
a4_tr = rs4_tr/n4_tr
b4_tr = cs4_tr/n4_tr
er4_tr = 1-sum(diag4_tr)/n4_tr
ac4_tr = 1-er4_tr
pr4_tr = diag4_tr/cs4_tr
r4_tr = diag4_tr/rs4_tr
f14_tr = 2*pr4_tr*r4_tr/(pr4_tr+r4_tr)

#KNN - Testing Error different accuracy measures

n4_ts = sum(cm4_ts)
nc4_ts = nrow(cm4_ts)
diag4_ts = diag(cm4_ts)
rs4_ts = apply(cm4_ts, 1, sum)
cs4_ts = apply(cm4_ts, 2, sum)
a4_ts = rs4_ts/n4_ts
b4_ts = cs4_ts/n4_ts
er4_ts = 1-sum(diag4_ts)/n4_ts
ac4_ts = 1-er4_ts
pr4_ts = diag4_ts/cs4_ts
r4_ts = diag4_ts/rs4_ts
f14_ts = 2*pr4_ts*r4_ts/(pr4_ts+r4_ts)

#Arrays for DataFrame

name = c('Error Rate', 'Accuracy', 'No. of Instances', 'No. of classes', 'No. of correctly classified instances per class (Die)', 'No. of correctly classified instances per class (Live)', 'No. of instances per class (Die)', 'No. of instances per class (Live)', 'No. of predictions per class (Die)', 'No. of predictions per class (Live)', 'Distribution of Instances over the actual classes (Die)', 'Distribution of Instances over the actual classes (Live)', 'Distribution of Instances over the predicted classes (Die)', 'Distribution of Instances over the predicted classes (Live)', 'Precision (Die)', 'Precision (Live)', 'Recall (Die)', 'Recall (Live)', 'F1 Measure (Die)', 'F1 Measure (Live)')
svm_tr = c(er1_tr, ac1_tr, n1_tr, nc1_tr, unname(diag1_tr[1]), unname(diag1_tr[2]), unname(rs1_tr[1]), unname(rs1_tr[2]), unname(cs1_tr[1]), unname(cs1_tr[2]), unname(a1_tr[1]), unname(a1_tr[2]), unname(b1_tr[1]), unname(b1_tr[2]), unname(pr1_tr[1]), unname(pr1_tr[2]), unname(r1_tr[1]), unname(r1_tr[2]), unname(f11_tr[1]), unname(f11_tr[2]))
svm_ts = c(er1_ts, ac1_ts, n1_ts, nc1_ts, unname(diag1_ts[1]), unname(diag1_ts[2]), unname(rs1_ts[1]), unname(rs1_ts[2]), unname(cs1_ts[1]), unname(cs1_ts[2]), unname(a1_ts[1]), unname(a1_ts[2]), unname(b1_ts[1]), unname(b1_ts[2]), unname(pr1_ts[1]), unname(pr1_ts[2]), unname(r1_ts[1]), unname(r1_ts[2]), unname(f11_ts[1]), unname(f11_ts[2]))
nb_tr = c(er2_tr, ac2_tr, n2_tr, nc2_tr, unname(diag2_tr[1]), unname(diag2_tr[2]), unname(rs2_tr[1]), unname(rs2_tr[2]), unname(cs2_tr[1]), unname(cs2_tr[2]), unname(a2_tr[1]), unname(a2_tr[2]), unname(b2_tr[1]), unname(b2_tr[2]), unname(pr2_tr[1]), unname(pr2_tr[2]), unname(r2_tr[1]), unname(r2_tr[2]), unname(f12_tr[1]), unname(f12_tr[2]))
nb_ts = c(er2_ts, ac2_ts, n2_ts, nc2_ts, unname(diag2_ts[1]), unname(diag2_ts[2]), unname(rs2_ts[1]), unname(rs2_ts[2]), unname(cs2_ts[1]), unname(cs2_ts[2]), unname(a2_ts[1]), unname(a2_ts[2]), unname(b2_ts[1]), unname(b2_ts[2]), unname(pr2_ts[1]), unname(pr2_ts[2]), unname(r2_ts[1]), unname(r2_ts[2]), unname(f12_ts[1]), unname(f12_ts[2]))
dt_tr = c(er3_tr, ac3_tr, n3_tr, nc3_tr, unname(diag3_tr[1]), unname(diag3_tr[2]), unname(rs3_tr[1]), unname(rs3_tr[2]), unname(cs3_tr[1]), unname(cs3_tr[2]), unname(a3_tr[1]), unname(a3_tr[2]), unname(b3_tr[1]), unname(b3_tr[2]), unname(pr3_tr[1]), unname(pr3_tr[2]), unname(r3_tr[1]), unname(r3_tr[2]), unname(f13_tr[1]), unname(f13_tr[2]))
dt_ts = c(er3_ts, ac3_ts, n3_ts, nc3_ts, unname(diag3_ts[1]), unname(diag3_ts[2]), unname(rs3_ts[1]), unname(rs3_ts[2]), unname(cs3_ts[1]), unname(cs3_ts[2]), unname(a3_ts[1]), unname(a3_ts[2]), unname(b3_ts[1]), unname(b3_ts[2]), unname(pr3_ts[1]), unname(pr3_ts[2]), unname(r3_ts[1]), unname(r3_ts[2]), unname(f13_ts[1]), unname(f13_ts[2]))
knn_tr = c(er4_tr, ac4_tr, n4_tr, nc4_tr, unname(diag4_tr[1]), unname(diag4_tr[2]), unname(rs4_tr[1]), unname(rs4_tr[2]), unname(cs4_tr[1]), unname(cs4_tr[2]), unname(a4_tr[1]), unname(a4_tr[2]), unname(b4_tr[1]), unname(b4_tr[2]), unname(pr4_tr[1]), unname(pr4_tr[2]), unname(r4_tr[1]), unname(r4_tr[2]), unname(f14_tr[1]), unname(f14_tr[2]))
knn_ts = c(er4_ts, ac4_ts, n4_ts, nc4_ts, unname(diag4_ts[1]), unname(diag4_ts[2]), unname(rs4_ts[1]), unname(rs4_ts[2]), unname(cs4_ts[1]), unname(cs4_ts[2]), unname(a4_ts[1]), unname(a4_ts[2]), unname(b4_ts[1]), unname(b4_ts[2]), unname(pr4_ts[1]), unname(pr4_ts[2]), unname(r4_ts[1]), unname(r4_ts[2]), unname(f14_ts[1]), unname(f14_ts[2]))

df = data.frame(Parameters=name, SVM_Train=svm_tr, SVM_Test=svm_ts, NaiveBayes_Train=nb_tr, NaiveBayes_Test=nb_ts, DecisionTree_Train=dt_tr, DecisionTree_Test=dt_ts, KNN_Train=knn_tr, KNN_Test=knn_ts)

write.csv(df, file="/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/Results/all_features8.csv", row.names = FALSE)