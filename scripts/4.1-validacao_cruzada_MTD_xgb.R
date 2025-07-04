library(caret)
library(tidyverse)
library(tidytext)
library(dplyr)
library(textstem)
library(xgboost)
library(pROC)
library(tm)

data(stop_words)




set.seed(123456789)
base = readRDS("base_completa_final.RDS")

dim(base)
colnames(base)

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
  
  indices_treino = NULL
  for(j in 1:k){
    if(j != i){
      indices_treino = c(indices_treino,folds[[j]])
    }
  }
  base_treino = base[indices_treino,]   
  
  
  #1) com a base de treino fazer o pre processamento de texto
  # tokenização, lematizacao, stops woords, 
  # selecionar os 300 mais freq e montar a matriz termo documento
  #
  
  base_treino = 
    base_treino |> 
    select(indices,resposta,rotulo)

  
  base_treino$resposta = base_treino$resposta |> removeNumbers()
  base_treino$resposta = base_treino$resposta |> removePunctuation() 
  
  base_treino_tokens = 
    base_treino |>
    unnest_tokens(output = word, 
                  input = resposta)
  
  base_treino_tokens = 
    base_treino_tokens |> 
    anti_join(stop_words)
  
  lemma_word = 
    lemmatize_words(
      base_treino_tokens$word)
  
  base_treino_tokens$word = 
    lemma_word
  
  base_treino_tokens = 
    base_treino_tokens |> 
    anti_join(stop_words)
  
  count_word =
    base_treino_tokens |>
    count(word, sort = TRUE) 
  dim(count_word)
  
  words_in = count_word[1:dimen,"word"]
  words_in = words_in |> arrange(word)
  words_in$word[1:10]
  
  base_treino_tokens = 
    base_treino_tokens |> 
    right_join(words_in)
  
  dim(base_treino_tokens)
  base_treino_tokens = 
    base_treino_tokens |> 
    select(-rotulo)
  #matriz termo docuemto
  
  base_treino_tokens_cont = 
    base_treino_tokens |> 
    group_by(indices,word) |> 
    mutate(n = n())
    
  mtd_treino = 
    base_treino_tokens_cont |> 
    cast_dfm(
      term = word,
      document = indices,
      value = n)
  mtd_treino = 
    mtd_treino[,sort(colnames(mtd_treino))]
  
  X_treino = as.matrix(mtd_treino)
  
  Y_treino = base_treino |>
    filter(indices %in% as.numeric(row.names(X_treino))) |> 
    select(indices,rotulo)
  Y_treino_ = ifelse(Y_treino$rotulo=="chatgpt",1,0)
  names(Y_treino_) = Y_treino$indices
  Y_treino = Y_treino_
  
  #names(Y_treino)==rownames(X_treino)
  #names(Y_treino)[100:110]==rownames(X_treino)[100:110]
  
  xgb_100 = xgboost(
    data = X_treino,
    label = Y_treino,
    objective = "binary:logistic",
    nrounds = 100)
  
  y_treino = predict(
    object = xgb_100,
    newdata = X_treino)
  
  auc_treino = roc(
    response = Y_treino,
    predictor = y_treino)
  
  auc = as.numeric(auc_treino$auc)
  AUC_treino = c(AUC_treino,auc)
  
  soma =
    coords(auc_treino)$specificity + coords(auc_treino)$sensitivity
  p = coords(auc_treino)$threshold[which.max(soma)]
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
  
  
  
  
  base_teste = base[folds[[i]],]
  
  base_teste = base_teste |> 
    select(indices,resposta,rotulo)
  
  base_teste$resposta <-
    base_teste$resposta %>% 
    removeNumbers()
  
  base_teste$resposta <-
    base_teste$resposta %>% 
    removePunctuation()
  
  base_teste_tokens = 
    base_teste |>
    unnest_tokens(output = word, 
                  input = resposta)
  dim(base_teste_tokens)
  
  lemma_word = 
    lemmatize_words(
      base_teste_tokens$word)
  
  base_teste_tokens$word = 
    lemma_word
  
  #dim(base_teste_tokens)
  
  base_teste_tokens = 
    base_teste_tokens |> 
    right_join(words_in)
  #dim(base_teste_tokens)
  
  base_teste_tokens = 
    base_teste_tokens |> 
    select(-rotulo)
  
  
  #dim(base_teste_tokens)
  base_teste_contagem = 
    base_teste_tokens |> 
    group_by(indices,word) |> 
    mutate(n = n())
  
  #dim(base_teste_contagem)
  
  mtd_teste = base_teste_contagem |> 
    cast_dfm(
      term = word,
      document = indices,
      value = n)
  #dim(mtd_teste)
  mtd_teste = mtd_teste[,sort(colnames(mtd_teste))]
  
  X_teste = as.matrix(mtd_teste)
  X_teste = X_teste[,xgb_100$feature_names]
  
  Y_teste = base_teste |> 
    filter(indices %in% rownames(mtd_teste))
  Y_teste_ = ifelse(Y_teste$rotulo=="chatgpt",1,0)
  names(Y_teste_) = Y_teste$indices
  Y_teste = Y_teste_
  
  y_teste  = predict(
    object = xgb_100,
    newdata= X_teste)
  
  
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
        file = "resultados_treino_BoW.RDS")

resultados_teste_XGB_100 = 
  data.frame(
    AUC_teste,
    Acuracia_teste,
    Sensibilidade_teste,
    Especificidade_teste,
    Precisao_teste)

saveRDS(resultados_teste_XGB_100,
        file = "resultados_teste_BoW.RDS")






