library(caret)
library(tidyverse)
library(dplyr)
library(textstem)
library(tm)
library(xgboost)
library(pROC)
library(readr)
library(word2vec)
library(doc2vec)



base = readRDS("base_completa_final.RDS")
folds = readRDS("folds.RDS")

k = length(folds)

AUC_teste = NULL
Acuracia_teste = NULL
Sensibilidade_teste = NULL
Especificidade_teste = NULL
Precisao_teste = NULL

AUC_treino = NULL
Acuracia_treino = NULL
Sensibilidade_treino = NULL
Especificidade_treino = NULL
Precisao_treino = NULL

dimen = 725

for(i in 1:k){

  print(i)
  
  base_teste = base[folds[[i]],]
  
  indices_treino = NULL
  for(j in 1:k){
    if(j != i){
      indices_treino = c(
        indices_treino,folds[[j]])
    }
  }
  base_treino = base[indices_treino,]   
  
  base_treino$resposta = base_treino$resposta |> removeNumbers()
  base_treino$resposta = base_treino$resposta |> removePunctuation() 
  
  model_word2vec = word2vec(
    x= c(base_treino$resposta),
    type="cbow",
    dim=dimen)
  
  
  #saveRDS(model_word2vec,"model_word2vec_1.RDS")
  
  indices = which(base_treino$resposta=="")
  if(length(indices)>0){
    base_treino = base_treino[-indices,]
  }
  
  X = data.frame(
    doc_id = base_treino$indices,
    text = base_treino$resposta
  )

  model_doc2vec_treino = 
    word2vec::doc2vec(
      object = model_word2vec,
      newdata = X
    )
  
  X_treino = as.matrix(model_doc2vec_treino)
  Y_treino = ifelse(base_treino$rotulo=="chatgpt",1,0)
  
  xgb_100 = xgboost(
    data = X_treino,
    label = Y_treino,
    objective = "binary:logistic",
    nrounds = 100)
  
  y_treino = predict(
    object = xgb_100,
    newdata = X_treino)
  
  
  indices = which(base_teste$resposta == "")
  if(length(indices)>0){
    base_teste = base_teste[-indices,]
  }
  
  X = data.frame(
    doc_id = base_teste$indices,
    text = base_teste$resposta
  )
  
  model_doc2vec_teste = 
    word2vec::doc2vec(
      object = model_word2vec,
      newdata = X
    )
  
  X_teste = as.matrix(model_doc2vec_teste)
  Y_teste = ifelse(base_teste$rotulo=="chatgpt",1,0)

  y_teste  = predict(
    object = xgb_100,
    newdata= X_teste)
  
  auc_treino = roc(
    response = Y_treino,
    predictor = y_treino)
  
  AUC_treino = c(AUC_treino,
                 as.numeric(auc_treino$auc))
  
  soma = 
    coords(auc_treino)$specificity + coords(auc_treino)$sensitivity
  i = which.max(soma)
  p = coords(auc_treino)$threshold[i]
  prev_classe_treino = as.factor(ifelse(y_treino < p,0,1))
  real_classe_treino = as.factor(Y_treino)
  
  CM = confusionMatrix(data = prev_classe_treino,
                       reference = real_classe_treino,
                       dnn = c("Previsto", "Real") )
  
  
  Acuracia_treino = c(Acuracia_treino,
                      CM$overall["Accuracy"])
  
  Sensibilidade_treino = c(Sensibilidade_treino,
                           CM$byClass["Sensitivity"])
  
  Especificidade_treino = c(Especificidade_treino,
                            CM$byClass["Specificity"])
  
  
  Precisao_treino = c(Precisao_treino,CM$byClass["Precision"])
  
  
  auc_teste = roc(
    response = Y_teste,
    predictor = y_teste)
  
  AUC_teste = c(AUC_teste,
                as.numeric(auc_teste$auc))
  
  prev_classe_teste = 
    as.factor(ifelse(y_teste < p,0,1))
  real_classe_teste = 
    as.factor(Y_teste)
  
  CM = confusionMatrix(
    data = prev_classe_teste,
    reference = real_classe_teste,
    dnn = c("Previsto", "Real") )
  
  
  Acuracia_teste = c(Acuracia_teste,
                     CM$overall["Accuracy"])
  
  Sensibilidade_teste = c(Sensibilidade_teste,
                          CM$byClass["Sensitivity"])
  
  Especificidade_teste = c(Especificidade_teste,
                           CM$byClass["Specificity"])
  
  
  
  Precisao_teste = c(Precisao_teste,CM$byClass["Precision"])
}



resultados_treino_XGB_100 = 
  data.frame(
    AUC_treino,
    Acuracia_treino,
    Sensibilidade_treino,
    Especificidade_treino,
    Precisao_treino)

saveRDS(resultados_treino_XGB_100,
        file = "resultados_treino_doc2vec.RDS")


resultados_teste_XGB_100 = 
  data.frame(
    AUC_teste,
    Acuracia_teste,
    Sensibilidade_teste,
    Especificidade_teste,
    Precisao_teste)

saveRDS(resultados_teste_XGB_100,
        file = "resultados_teste_doc2vec.RDS")
