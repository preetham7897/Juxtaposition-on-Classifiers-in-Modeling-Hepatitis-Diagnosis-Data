#importing libraries

library('VIM')
library('mice')

#importing data

data = read.csv('/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/data.csv', header=TRUE)

#Imputing Missing Values

imp = mice(data)
data = complete(imp)

#Converting to csv

write.csv(data, file="/home/preetham/Documents/Research Projects/Hepatitis Classification/Data/Imputed Data.csv", row.names = FALSE)