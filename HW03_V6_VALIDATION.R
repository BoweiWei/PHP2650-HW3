########################################################################## Author list and packages list
authors <- function(){
  c("Bernard Chu", "Bowei Wei", "Jiarou Quan", "Ning Zhang")
}

require(RSQLite) == T || install.packages("RSQLite", repos="https://cloud.r-project.org/")
require(glmnet) == T || install.packages("glmnet", repos="https://cloud.r-project.org/")
require(dplyr) == T || install.packages("RSQLite", repos="https://cloud.r-project.org/")
require(data.table) == T || install.packages("data.table", repos="https://cloud.r-project.org/")
require(sparklyr) == T || install.packages("sparklyr", repos="https://cloud.r-project.org/")
require(rsparkling) == T || install.packages("rsparkling", repos="https://cloud.r-project.org/")
require(h2o) == T || install.packages("h2o", repos="https://cloud.r-project.org/")

########################################################################## Question 1
## Input sqlite file and extract tables
db <- dbConnect(SQLite(), dbname="pred.sqlite")
tables = dbListTables(db)

# exclude sqlite_sequence (contains table information)
tables_names <- tables[tables != "sqlite_sequence"]
mytables <- lapply(tables_names, dbReadTable, conn = db) 
mytables[[1]][,2:3] <- lapply(mytables[[1]][,2:3], as.factor)

# inner join the 3 tables to get the common ids (with predictors)
mytables %>%
  Reduce(function(dtf1,dtf2) inner_join(dtf1,dtf2,by="id"), .) -> train_comp

# missing ids (with predictors)
idmissing_data <- anti_join( inner_join(mytables[[1]], mytables[[3]]),
                             mytables[[2]])
idall <- data.frame(id = train_comp$id)
idmissing <- data.frame(id = idmissing_data$id)

write.csv(idall ,"idall.csv", row.names=FALSE)
write.csv(idmissing ,"idmissing.csv",row.names=FALSE)

########################################################################## Question 2 and 3
# remove outcome variables
train <- train_comp[,-c(4:21)]

# create dummy variables
train_mtx <- model.matrix(~ ., train[, -1])[, -1]
test_mtx <- model.matrix(~ ., idmissing_data[, -1])[, -1]

# labels
y <- train_comp[, 4:21]

lasso_predict <- function(train, label, newx, 
                          lambda_grid = 10^seq(-3,3, length=100) ){
  # function to cross validate lasso models to determine optimal lambda,
  # predict labels with the optimal model
  # output a list with predicted values along with cross validation error
  cv_lasso <- cv.glmnet(train, label, lambda = lambda_grid, alpha =1)
  lasso_fit <- glmnet(train, label, lambda = cv_lasso$lambda.min, alpha =1)
  output <- predict(lasso_fit, newx , type = "response")
  output_cvm <- min(cv_lasso$cvm)
  list(output = output, output_cvm = output_cvm)
}


system.time(missingID_pred <- sapply( X=y, lasso_predict, train= train_mtx, newx = test_mtx))

# sum all cv error from the 18 models
cvm_pred1 <- sum(unlist(missingID_pred[seq(2,36,2)])) 

# output predicted values
output2 <- data.frame(id = idmissing, do.call(cbind,missingID_pred[seq(1,35,2)]))
colnames(output2)[-1] = paste("o",1:18, sep="")
write.csv(output2[ ,1:2], "output1.csv",row.names=FALSE)
write.csv(output2, "output2.csv",row.names=FALSE)

# user  system elapsed 
# 6.035   0.013   6.047

########################################################################## Question 4
pred2 <- read.csv("pred2.csv", header = FALSE, sep = ",")
colnames(pred2) <- c("id", paste("V",106:304, sep =""))
train_mtx2 <- inner_join(data.frame(id = train$id, train_mtx), pred2, by = "id")[ ,-1]
test_mtx2 <- inner_join(data.frame(id = idmissing_data$id, test_mtx), pred2, by = "id")[ ,-1]

# train lasso models with additional predictors
system.time(missingID_pred2 <- sapply( X=y, lasso_predict, train= as.matrix(train_mtx2), newx = as.matrix(test_mtx2)))

# sum all cv error from the 18 models again
cvm_pred2 <- sum(unlist(missingID_pred2[seq(2,36,2)]))

output3 <- data.frame(id = idmissing, do.call(cbind,missingID_pred2[seq(1,35,2)]))
colnames(output3)[-1] = paste("o",1:18, sep="")
cat(paste("cv error of less predictors:",cvm_pred1,  "\ncv error of more predictors:", cvm_pred2))
write.csv(output3, "output3.csv",row.names=FALSE)

## There is no significant difference after adding 199 more predictors from the results above (a little bit improvement). More data will help to improve
## the prediction performance. But the improvement has marginal utility. The effective improvement introduced by additional data will
## decrease as data increase. Add small amount of data like pred2.csv provided won't be very helpful.

# user  system elapsed 
# 155.827   0.000 155.800

########################################################################## Question 5
library(data.table)
pred3 = fread("pred3.csv")
colnames(pred3) <- c("id", paste("V",305:45154, sep =""))
train_mtx3 <- inner_join(data.frame(id = train$id, train_mtx), pred3, by = "id")[ ,-1]
test_mtx3 <- inner_join(data.frame(id = idmissing_data$id, test_mtx), pred3, by = "id")[ ,-1]

# train lasso models with additional large amount of predictors
system.time(missingID_pred3 <- sapply( X=y, lasso_predict, train= as.matrix(train_mtx3), newx = as.matrix(test_mtx3)))

# sum all cv error from the 18 models again
cvm_pred3 <- sum(unlist(missingID_pred3[seq(2,36,2)]))

output4 <- data.frame(id = idmissing, do.call(cbind,missingID_pred3[seq(1,35,2)]))
colnames(output4)[-1] = paste("o",1:18, sep="")
cat(paste("cv error of less predictors:",cvm_pred1,  "\ncv error of more predictors:", cvm_pred2), "\ncv error of additional large amount of predictors:", cvm_pred3)
write.csv(output4, "output4.csv",row.names=FALSE)

##user       system      elapsed 
##1911.785   56.811      1968.117

##cv error of less predictors: 2490.06605570536 
##cv error of more predictors: 2482.84608343852 
##cv error of additional large amount of predictors: 2495.131

########################################################################## Question 4 SPARK VALIDATION AND VISUALIZATIOND 
library(sparklyr)
library(data.table)
library(dplyr)
library(rsparkling)
library(h2o)
library(ggplot2)

pred2 <- read.csv("pred2.csv", header = FALSE, sep = ",")
colnames(pred2) <- c("id", paste("V",106:304, sep =""))
train_mtx2 <- inner_join(data.frame(id = train$id, train_mtx), pred2, by = "id")[ ,-1]
test_mtx2 <- inner_join(data.frame(id = idmissing_data$id, test_mtx), pred2, by = "id")[ ,-1]
train_y <- cbind(y,train_mtx2) 

spark_conn <- spark_connect(master="spark://slbd2018-new.c.slbd-nz.internal:7077", spark_home="/usr/lib/spark/")
spark_version(spark_conn)
data_tbl <- copy_to(spark_conn, train_y, overwrite =TRUE)
set.seed(0) # for reproducible results. Any number is good.

src_tbls(spark_conn)
data_tbl <- tbl(spark_conn, "train_y")
glimpse(data_tbl)
object.size(data_tbl)

# ## ML in Sparklyr: ml_LR can only handle single response and single lambda at each time. Try to use other method
# lasso_model<- data_tbl %>%
#   ml_linear_regression(response="O1", features = c("genderM","age26","age31","age36",paste("V",1:304, sep ="")), elastic_net_param = 1)
# ## alpha=1, LASSO Regression. 
# ## Cross validation in Sparklyr
# ml_cross_validator()

## Try to use Rsparkling
partitions <- data_tbl %>%
  sdf_partition(training=0.8, testing= 0.2)
training<- as_h2o_frame(spark_conn,partitions$training,strict_version_check=FALSE)
testing<- as_h2o_frame(spark_conn,partitions$testing,strict_version_check=FALSE)

lasso_model_h2o <- h2o.glm(y="O1", x=c("genderM","age26","age31","age36",paste("V",1:304, sep ="")), training_frame=training, lambda_search= TRUE, alpha=1,ignore_const_cols = FALSE)
show(lasso_model_h2o)

pred <- h2o.predict(lasso_model_h2o, newdata = testing[,c("genderM","age26","age31","age36",paste("V",1:304, sep =""))])

## visulization and validation
predicted <- as_spark_dataframe(spark_conn, pred, strict_version_check = FALSE)
actual <- partitions$test %>%
  select(O1) %>%
  collect()

data <- data.frame(
  predicted = predicted,
  actual    = actual
)

names(data) <- c("predicted", "actual")
ggplot(data, aes(x = actual, y = predicted)) +
  geom_abline(lty = "dashed", col = "red") +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Actual Fuel Consumption",
    y = "Predicted Fuel Consumption",
    title = "Predicted vs. Actual Fuel Consumption"
  )