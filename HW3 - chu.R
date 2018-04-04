library(RSQLite)
library(dplyr)
library(glmnet)
db <- dbConnect(SQLite(), dbname="pred.sqlite")
tables = dbListTables(db)

## exclude sqlite_sequence (contains table information)
tables_names <- tables[tables != "sqlite_sequence"]
mytables <- lapply(tables_names, dbReadTable, conn = db) 
demo <- mytables[[1]]
outcome <- mytables[[2]]
pred <- mytables[[3]]

mytables %>%
  Reduce(function(dtf1,dtf2) inner_join(dtf1,dtf2,by="id"), .) -> train_comp

idmissing <- setdiff(train$id, outcome$id)
idall <- train$id

###write csv here
train_comp[,2:3] <- lapply(train_comp[,2:3], as.factor)
train <- train_comp[,-c(4:21)]


train_mtx <- model.matrix(~ ., train[, -1])[, -1]
y <- as.matrix(train_comp[, 4:21])

cv_lasso <- cv.glmnet(as.matrix(train_mtx), as.matrix(y,ncol=1), family = "mgaussian", nlambda = 200,alpha =1)
plot(cv_lasso)

