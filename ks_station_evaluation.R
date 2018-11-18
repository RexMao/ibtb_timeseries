library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(abind)
library(forecast)
source('gpusetting.R')

# load data
station05_17 <- fread('train0517.csv')
station18 <- fread('train18.csv')

# choose kaohsiung station
ks_train <- station05_17[station05_17$TKT_BEG==185,]
ks_test <- station18[station18$TKT_BEG==185,]

# preprocessing
ks_train$BOARD_DATE = as.Date(as.character(ks_train$BOARD_DATE),'%Y%m%d')
ks_train$year = factor(year(ks_train$BOARD_DATE))

# data visualize
ggplot(ks_train, aes(x=year, y=進站)) + geom_boxplot()

train <- ks_train$進站
test <- ks_test$進站

# normalize data
mean <- mean(train)
std <- mean(train)
train <- scale(train, center = mean, scale = std)
test <- scale(test, center = mean, scale = std)


# load training & testing data
# do not use this method when data is large
traindata <- data.frame()
trainlabel <- c()
for (i in c(21:4748)){
  traindata <- rbind(traindata, train[(i-20):(i-1)])
  trainlabel <- c(trainlabel, train[i])
}

testdata <- data.frame()
testlabel <- c()
for (i in c(21:181)){
  testdata <- rbind(testdata, test[(i-20):(i-1)])
  testlabel <- c(testlabel, test[i])
}



traindata <- as.matrix(traindata)
trainlabel <- as.matrix(trainlabel)
testdata <- as.matrix(testdata)
testlabel <- as.matrix(testlabel)


train <- array_reshape(traindata, dim = c(dim(traindata)[[1]],1,dim(traindata)[[2]]))
test <- array_reshape(testdata, dim = c(dim(testdata)[[1]],1,dim(testdata)[[2]]))


dim(train)
dim(test)

model <- keras_model_sequential() %>%
  layer_lstm(units = 64, return_sequences = TRUE,
             input_shape = c( 1,20 ) )%>%
  layer_lstm(units = 64, activation = "relu")%>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse",
  metrics = c("mae")
)

model %>% fit(train, trainlabel,
              epochs = 50, batch_size = 32, validation_split = 0.1)

# evaluation
result <- model %>% predict(test, verbose=0)
mean(abs(result - testlabel))


# plot result & store as a csv file
ks_mae <- mean(abs((result * std + mean) - (testlabel* std + mean)))

dat <- data.frame(c=0:160, a=result, b=testlabel)

dat %>%
  gather(key,value, a, b) %>%
  ggplot(aes(x=c, y=value, colour=key)) +
  geom_line() +
  geom_point() 


ks_result <- cbind(testlabel* std + mean, result * std + mean)

write_csv(ks_result, "ks_result.csv")


# predict N step
window_size = 20
step = 10
res <- data.frame()
start_idx <- 1
end_idx <- window_size
for (j in 1:length(b)-window_size-step){
  temp <- c()
  data <- array_reshape(b[c(start_idx:end_idx)], dim = c(1,1,20))
  for (i in 1:step){
    preds <- model %>% predict(data, verbose=0)
    mae <- abs((preds* std + mean) - (b[end_idx+i]* std + mean))
    temp <- c(temp, mae)
    data <- abind(data, preds)
    data <- array_reshape(data[1,1,2:21], dim = c(1,1,20))
  }
  res <- rbind(res, temp)
  start_idx <- start_idx + 1
  end_idx <- end_idx + 1
}

as.vector(sapply(res, mean, na.rm=TRUE))

# N step plot
dat <- data.frame('step'=1:10, 'MSE'=as.vector(sapply(res, mean, na.rm=TRUE)))

dat %>%
  ggplot(aes(x=step, y=MSE)) +
  geom_line() +
  geom_point() 


# auto arima
arima <- auto.arima(a, seasonal=F)

fcast_no_holdout <- forecast(arima,h=181)
fcast_no_holdout$mean
mean(abs(fcast_no_holdout$mean - b$a))

# garch
library(rugarch)
model <- ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1,1)),mean.model = list(armaOrder=c(4,2)),distribution.model="std")
egarch <- ugarchfit(a$a, spec=model)
egarch

gmodel = ugarchforecast(egarch, n.ahead=181)

mean(abs(gmodel@forecast$seriesFor - b$a))

# prophet
library(prophet)

pdata <- data.frame(ds=as.Date("2005-1-1 0:00")+0:4747,y=a$a)


m <- prophet(pdata)
future <- make_future_dataframe(m, periods = 181, freq = 'day', include_history = F)

forecast <- predict(m, future)

mean(abs(forecast$yhat - b$a))

