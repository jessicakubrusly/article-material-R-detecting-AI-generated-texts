library(tidyverse)

base = readRDS("base_completa.RDS")

N = length(base$question)


resposta = NULL
pergunta = NULL
rotulo = NULL
fonte = NULL

for(i in 1:N){
  
  if(length(base$chatgpt_answers[[i]]) == 1){
    
    pergunta = c(pergunta,base$question[i])
    resposta = c(resposta,base$chatgpt_answers[[i]])
    rotulo = c(rotulo,"chatgpt")
    fonte = c(fonte,base$source[i])
    
    #quantos humanos responderam essa pergunta
    k = length(base$human_answers[[i]])
    for(j in 1:k){
      pergunta = c(pergunta,base$question[i])
      resposta = c(resposta,base$human_answers[[i]][j])
      rotulo = c(rotulo,"humano")
      fonte = c(fonte,base$source[i])
    }
  }
}

length(pergunta)
length(resposta)
length(rotulo)
table(rotulo)
length(fonte)

n = length(fonte)

base_completa_final = data.frame(indices = (1:n), 
                                 pergunta,
                                 resposta,
                                 rotulo,
                                 fonte)

base_completa_final = tibble(base_completa_final)

saveRDS(base_completa_final,file = "base_completa_final.RDS")
