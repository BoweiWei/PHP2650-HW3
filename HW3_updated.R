########################################################################## Author list and packages list
authors <- function(){
  c("Bernard Chu", "Bowei Wei", "Jiarou Quan", "Ning Zhang")
}

require(RSQLite) == T || install.packages("RSQLite", repos="https://cloud.r-project.org/")
require(glmnet) == T || install.packages("glmnet", repos="https://cloud.r-project.org/")
require(dplyr) == T || install.packages("RSQLite", repos="https://cloud.r-project.org/")

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
                          lambda_grid = 10^seq(-2,2, length=100) ){
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
