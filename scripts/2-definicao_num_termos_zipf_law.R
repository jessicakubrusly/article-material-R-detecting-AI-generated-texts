library(tidyverse)
library(tidytext)
library(textstem)
library(tm)



base_treino = readRDS("base_completa_final.RDS")
glimpse(base_treino)


base_treino = 
  base_treino |> 
  select(indices,resposta,rotulo)

base_treino$resposta = base_treino$resposta |> removeNumbers()
base_treino$resposta = base_treino$resposta |> removePunctuation() 

base_treino_tokens = 
  base_treino |>
  unnest_tokens(output = word, 
                input = resposta)

lemma_word = 
  lemmatize_words(
    base_treino_tokens$word)

base_treino_tokens$word = 
  lemma_word


count_word = base_treino_tokens |> 
  select(word) |> 
  group_by(word) |>  
  mutate(n = n()) |> 
  ungroup() |>
  unique() |>
  arrange(desc(n)) |> 
  mutate(rank = row_number())


lm(log(count_word$n[1:500]) ~ 
     log(count_word$rank[1:500]))

windows(100,50)
plot(x=count_word$rank, 
     y=count_word$n,log = "xy",
     xlab="log(rank)",ylab="log(freq)",
     cex.lab=1.5,type="l",lwd=2,axes=F)
box()
axis(2,at = c(1,10,100,1000,10000),cex.axis = 1.5)
axis(1,at = c(1,10,100,1000,10000),cex.axis = 1.5)
curve(exp(14-log(x)),add = T,col="red",lwd=2)
abline(v=1000,lty=2,col="gray",lwd=2)
abline(h=1000,lty=2,col="gray",lwd=2)

count_word = count_word[1:1000,]
dim(count_word)

data(stop_words)

count_word = count_word |> 
  anti_join(stop_words)

dim(count_word)
