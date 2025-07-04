library(tidyverse)
library(caret)

base = readRDS("base_completa_final.RDS")

glimpse(base)
sum(is.na(base$resposta))
dim(base)

N = nrow(base)

set.seed(1357902468)
folds = createFolds(y=base$rotulo, 
                    k = 10, 
                    list = TRUE, 
                    returnTrain = FALSE)
class(folds)
length(folds)
class(folds[[1]])
folds[[1]]

table(base$rotulo)

table(base$rotulo[folds[[1]]])
table(base$rotulo[folds[[2]]])
table(base$rotulo[folds[[3]]])


saveRDS(folds,"folds.RDS")
