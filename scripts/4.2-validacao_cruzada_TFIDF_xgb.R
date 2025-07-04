library(caret)
library(tidyverse)
library(tidytext)
library(dplyr)
library(textstem)
library(xgboost)
library(pROC)
library(tm)
library(janeaustenr)


data(stop_words)


set.seed(1234567890)

#### aqui comeca o codigo de vcs

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
  
  words_in = count_word[1:dimen,"word"]
  words_in = words_in |> arrange(word)
  
  base_treino_tokens = 
    base_treino_tokens |> 
    right_join(words_in)
  
  base_treino_tokens = 
    base_treino_tokens |> 
    select(-rotulo)
  
  
  #TF-IDF
  
  
  base_treino_tokens = 
    base_treino_tokens |> 
    count(indices, word, sort = TRUE)
  
  total_words =
    base_treino_tokens |> 
    group_by(indices) |> 
    summarize(total = sum(n))
  
  base_treino_tokens = 
    base_treino_tokens |> 
    left_join(total_words)
  
  base_treino_tokens_tfidf = 
    base_treino_tokens |> 
    bind_tf_idf(word,indices,n) |>
    arrange(desc(total))
  
  #matriz termo docuemto
  
  mtd_treino_tfidf = 
    base_treino_tokens_tfidf |> 
    cast_dfm(
      term = word,
      document = indices,
      value = tf_idf)

  X_treino = as.matrix(mtd_treino_tfidf)
  
  Y_treino = base_treino |>
    filter(indices %in% as.numeric(row.names(X_treino))) |> 
    select(indices,rotulo)
  Y_treino_ = ifelse(Y_treino$rotulo == "chatgpt",1,0) 
  names(Y_treino_) = Y_treino$indices
  Y_treino_ = Y_treino_[rownames(X_treino)]
  Y_treino = Y_treino_
  # names(Y_treino)[290:300]
  # rownames(X_treino)[290:300]
  
  xgb_100 = xgboost(
    data = X_treino,
    label = Y_treino,
    objective = "binary:logistic",
    nrounds = 100)
  
  y_treino = predict(
    object = xgb_100,
    newdata = X_treino)
  
  
  
  base_teste = base[folds[[i]],]
  
  base_teste = 
    base_teste |> 
    select(indices,resposta,rotulo)
  
  
  base_teste$resposta = base_teste$resposta |> removeNumbers()
  base_teste$resposta = base_teste$resposta |> removePunctuation() 
  
  
  base_teste_tokens = 
    base_teste |>
    unnest_tokens(output = word, 
                  input = resposta)
  
  base_teste_tokens = 
    base_teste_tokens |> 
    anti_join(stop_words)
  
  lemma_word = 
    lemmatize_words(
      base_teste_tokens$word)
  
  base_teste_tokens$word = 
    lemma_word
  
  base_teste_tokens = 
    base_teste_tokens |> 
    anti_join(stop_words)
  
  
  base_teste_tokens = 
    base_teste_tokens |> 
    right_join(words_in)
  
  base_teste_tokens = 
    base_teste_tokens |> 
    select(-rotulo)
  
  
  idf = base_treino_tokens_tfidf |> select(word,idf)
  idf = unique(idf)
  
  words_by_indices = 
    base_teste_tokens |> 
    group_by(indices) |> 
    count() |> 
    rename(total = n)
  
  count_terms_in_indices = 
    base_teste_tokens |> 
    count(word,indices) 
  base_teste_tokens_tfidf = 
  base_teste_tokens |> 
    inner_join(count_terms_in_indices) |> 
    inner_join(words_by_indices) |> 
    mutate(tf = n/total) |> 
    inner_join(idf) |> 
    mutate(tfidf = tf*idf)
  
  
  mtd_teste_tfidf = 
    base_teste_tokens_tfidf |> 
    cast_dfm(
      term = word,
      document = indices,
      value = tfidf)
  
  X_teste = as.matrix(mtd_teste_tfidf)
  X_teste = X_teste[,colnames(X_treino)]
  
  Y_teste = base_teste |>
    filter(indices %in% as.numeric(row.names(X_teste))) |> 
    select(indices,rotulo)
  Y_teste_ = ifelse(Y_teste$rotulo == "chatgpt",1,0) 
  names(Y_teste_) = Y_teste$indices
  Y_teste_ = Y_teste_[rownames(X_teste)]
  Y_teste = Y_teste_
  # names(Y_teste)[290:300]
  # rownames(X_teste)[290:300]
  
  y_teste  = predict(
    object = xgb_100,
    newdata= X_teste)
  
  auc_treino = roc(
    response = Y_treino,
    predictor = y_treino)
  
  AUC_treino = c(AUC_treino,
                 as.numeric(auc_treino$auc))
  
  soma = coords(auc_treino)$specificity + coords(auc_treino)$sensitivity
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

resultados_treino_XGB_100_TFIDF = 
data.frame(
  AUC_treino,
  Acuracia_treino,
  Sensibilidade_treino,
  Especificidade_treino,
  Precisao_treino)

saveRDS(resultados_treino_XGB_100_TFIDF,
        file = "resultados_treino_TFIDF.RDS")


resultados_teste_XGB_100_TFIDF = 
  data.frame(
    AUC_teste,
    Acuracia_teste,
    Sensibilidade_teste,
    Especificidade_teste,
    Precisao_teste)

saveRDS(resultados_teste_XGB_100_TFIDF,
        file = "resultados_teste_TFIDF.RDS")

