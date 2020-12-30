# Projeto com Feedback 2 - Prevendo Demanda de Estoque com Base em Vendas

# Formação Cientista de Dados
# Módulo: Big Data Analitycs com R e Microsoft Azure Machine Learning
# Data Science Academy

# http://wwww.datascienceacademy.com.br/

# Os datasets usados e gerados durante a execução desse script estão disponíveis em:
# https://drive.google.com/drive/folders/1jwaQ2Ih696KMEimQOxksaAdqUxZQVxzo?usp=sharing

## Setando o diretório de trabalho
setwd("C:/Users/victor/Desktop/FDC/BDARMA/XVIII.Projetos-com-Feedback/02.Prevendo-Demanda-Estoque/BimboGroup_DemandPrediction/")
getwd()

library(tidyverse)
library(data.table)
library(ggplot2)
library(hrbrthemes)
library(fastDummies)
library(xgboost)
library(readr)

memory.limit(9999999999999)

## Lendo os dados
train <- fread("train.csv")
test <- fread("test.csv")

## Exploração dos dados
unique(train$Semana)
length(unique(train$Agencia_ID))
unique(train$Canal_ID)
length(unique(train$Ruta_SAK))
length(unique(train$Cliente_ID))
length(unique(train$Producto_ID))

train %>% 
  group_by(Semana) %>% 
  summarise(total = sum(Demanda_uni_equil)) %>% 
  arrange(-total) %>% 
  ggplot(aes(x = factor(Semana), y = total, fill = factor(Semana))) +
  geom_bar(stat = "identity",
           show.legend = FALSE) +
  labs(title = "Vendas Semanais",
       x = "Semana",
       y = "Vendas") +
  theme_tinyhand(base_size = 12)

# Não há grande diferença nas vendas de uma semana para a outra.

train %>% 
  group_by(Agencia_ID) %>% 
  summarise(soma = sum(Demanda_uni_equil)) %>% 
  arrange(-soma) %>% 
  ggplot(aes(x = soma)) +
  geom_boxplot(outlier.colour = "blue") +
  labs(title = "BoxPlot de Vendas Agência", 
       x = "Vendas") +
  theme_tinyhand(base_size = 12)

train %>% 
  group_by(Canal_ID) %>% 
  summarise(soma = sum(Demanda_uni_equil)) %>% 
  arrange(-soma) %>% 
  ggplot(aes(x = factor(Canal_ID), y = soma, fill = factor(Canal_ID))) +
  geom_bar(stat = "identity",
           show.legend = FALSE) +
  labs(title = "Vendas por Canal ID",
       x = "Canal",
       y = "Vendas") +
  theme_tinyhand(base_size = 12)

# Vemos que o Canal influencia na Demanda

boxplot((train %>% 
           group_by(Ruta_SAK) %>% 
           summarise(soma = sum(Demanda_uni_equil)))[,2], 
        horizontal = TRUE)
title("BoxPlot de Vendas por Rota")

train %>% 
  group_by(Cliente_ID) %>% 
  summarise(soma = sum(Demanda_uni_equil)) %>% 
  ggplot(aes(x = soma)) +
  geom_boxplot(outlier.colour = "blue") +
  labs(title = "BoxPlot de Vendas Por Cliente",
       x = "Vendas") +
  theme_tinyhand(base_size = 12)

# Tem um cliente que compra muito.

train %>% 
  group_by(Producto_ID) %>% 
  summarise(soma = sum(Demanda_uni_equil)) %>% 
  ggplot(aes(x = soma)) +
  geom_boxplot(outlier.color = "blue") +
  labs(title = "BoxPlot Vendas por Produto",
       x = "Vendas") +
  theme_tinyhand(base_size = 12)

## Função para extrair os Outliers com maiores demandas
outlier_categorical <- function(data, col){
  require(dplyr)
  
  filting <- data %>% 
    group_by_at(col) %>% 
    summarise(soma = sum(Demanda_uni_equil)) %>% 
    arrange(-soma)
  
  val.outliers <- boxplot.stats(filting$soma)$out
  outliers <- unlist((filting %>% 
                        filter(soma %in% val.outliers))[,1])
  names(outliers) <- NULL
  
  return(
    data %>% 
    mutate_at(c(val=col), as.integer) %>% 
    filter(!val %in% outliers) %>% 
    select(-val)
  )
}

# Usando Funçap para retirar os maiores compradores 
# que foram vistos no BoxPlot de Client_ID por Demanda
train <- outlier_categorical(data = train, col = "Cliente_ID")

## Retirando os Outliers da Variável Demanda_uni_equil (target)
val.outliers <- boxplot.stats(train$Demanda_uni_equil, coef = 4.5)$out

train <- train %>% 
  filter(!Demanda_uni_equil %in% val.outliers)

rm(val.outliers)
rm(outlier_categorical)

## Retirando dados que não serão utilizados
train$Venta_uni_hoy <- NULL
train$Venta_hoy <- NULL
train$Dev_uni_proxima <- NULL
train$Dev_proxima <- NULL

## Transformando Variável Canal_ID - Aplicando One Hot Encoding
train <- dummy_cols(.data = train, select_columns = "Canal_ID", remove_first_dummy = TRUE)
train$Canal_ID <- NULL
glimpse(train)

## Criando variáveis - Feature Engineering
train$Semana <- NULL
train$Ruta_SAK <- NULL
train$Cliente_ID <- NULL

train <- train %>% 
  group_by(Producto_ID) %>% 
  mutate(mean_prod = sum(Demanda_uni_equil)/n(),
         prop_prod = n()/nrow(train))

head(train[, c(12,13)])

train <- train %>% 
  group_by(Agencia_ID) %>% 
  mutate(mean_agen = sum(Demanda_uni_equil)/ n(),
         prop_agen = n()/nrow(train))

head(train[, c(14,15)])

## Normalizando variável target, possui muito valores 0
hist(train$Demanda_uni_equil, breaks = 10)
train$Demanda_uni_equil <- log1p(train$Demanda_uni_equil)
hist(train$Demanda_uni_equil, breaks = 10)

## Normalizando outra variáveis
train$mean_prod <- scale(train$mean_prod)[,1]
train$mean_agen <- scale(train$mean_agen)[,1]
train$prop_prod <- scale(train$prop_prod)[,1]
train$prop_agen <- scale(train$prop_agen)[,1]

## Separando Dataset dos Produtos, por Produto único
producto <- train[, c(2, 13, 12)] %>% 
  distinct()

train$Producto_ID <- NULL

## Separando Dataset das Agencias, por Agência única
agencia <- train %>% 
  select(Agencia_ID, mean_agen, prop_agen) %>% 
  distinct()

train$Agencia_ID <- NULL

## Transformando dataset de teste e preparando para previsão
glimpse(test)

test$Semana <- NULL
test$Ruta_SAK <- NULL
test$Cliente_ID <- NULL

test <- dummy_cols(.data = test, select_columns = "Canal_ID", remove_first_dummy = TRUE)

test <- test %>% 
  left_join(producto, by = "Producto_ID")

test <- test %>% 
  left_join(agencia, by = "Agencia_ID")

glimpse(test)

test$Agencia_ID <- NULL
test$Canal_ID <- NULL
test$Producto_ID <- NULL

# Liberando espaço na memória
write.csv(test, "test2.csv")

rm(agencia)
rm(producto)
rm(test)

## Dividindo Train em dataset de treino e teste
# Para testar e avaliar o modelo
glimpse(train)
setDT(train)

index <- sample(1:nrow(train), size = 0.7 * nrow(train), replace = FALSE)

logical <- vector()
length(logical) <- nrow(train)

for (i in index) {
  logical[i] <- 1
}

logical <- ifelse(is.na(logical) == TRUE, 0, 1)
train$index <- logical

# Removendo da Memória o que não será mais usado
rm(logical)
rm(index)
rm(i)

# Dividindo pelo Dplyr, pois é mais rápido que train[index,]
teste <- train %>% 
  filter(index == 0) %>% 
  select(-index)

train <- train %>% 
  filter(index == 1) %>% 
  select(-index)

## Treinando o modelo
# Salvando dataset de teste para liberar espaço e facilitar utilizações posteriores
# e liberar espaço na memória
write_csv(teste, "teste2.csv")
rm(teste)

labels <- train$Demanda_uni_equil
train <- train[, -1]

dtrain <- xgb.DMatrix(data = as.matrix(train),
                      label = labels, missing = NA)

rm(labels)
rm(train)

#write_rds(dtrain, "dtrain.rds")

model <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    booster = "gbtree",
    eta = 0.2,
    max_depth = 5,
    subsample = 0.7,
    colsample_bytree = 0.7
  ), data = dtrain,
  nrounds = 100,
  verbose = T,
  print_every_n = 5,
  maximize = FALSE,
  nthread = 16
)

#write_rds(model, "model.rds")

## Avaliando o Modelo
test1 <- read_csv("teste2.csv")

head(test1)
Y <- test1$Demanda_uni_equil
test1 <- test1 %>% select(-Demanda_uni_equil)

previsoes <- predict(model, as.matrix(test1))
previsoes <- expm1(previsoes)

sum(previsoes < 0)

for (i in 1:length(previsoes)) {
  if (previsoes[i] < 0) {
    previsoes[i] = 0
  }
}

sum(previsoes < 0)

actual <- expm1(Y)

error <- actual - previsoes
mse <- mean(error^2)
R2 = 1 - (sum(error^2)/sum((actual - mean(actual))^2))
## Prevendo no dataset oficial
test <- read_csv("test2.csv", col_types = cols("i","i", "i", "i", "i", "i", "i", "i", "i", "i",
                                                "d", "d","d", "d"))

test <- test %>% 
  relocate(X1,id,Canal_ID_2,Canal_ID_4,Canal_ID_5,Canal_ID_6,Canal_ID_7,Canal_ID_8,
           Canal_ID_9, Canal_ID_11, mean_prod, prop_prod, mean_agen, prop_agen) %>% 
  select(-X1)

previsoes.of <- predict(model, as.matrix(test %>% select(-id)))
previsoes.of <- expm1(previsoes.of)

sum(previsoes.of < 0)

# Como não existe demanda quebrada
previsoes.of <- round(previsoes.of)

## Criando Data Frame para submissão e salvando
df.submission <- data.frame(id = test$id,
                            Demanda_uni_equil = previsoes.of)

head(df.submission)
tail(df.submission)

write_csv(df.submission, "submission.csv")

## Após submissão nosso dataset de submissão obteve Score:
# 0.75518 Privado
# 0.74425 Público