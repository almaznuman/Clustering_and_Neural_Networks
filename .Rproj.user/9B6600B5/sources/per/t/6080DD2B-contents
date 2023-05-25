library(tidyverse)
library(neuralnet)
library(Metrics)
library(readxl)
library(keras)
#min-max normalization and unormalization
Normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
Unnormalize <- function(df, original_data){
  max_val <- max(original_data)
  min_val <- min(original_data)
  unnormalized_df <- df * (max_val - min_val) + min_val
  return(unnormalized_df)
}
# Neural network plot and statistical indices calculation
# Logistic and hyperbolic tangent activation function used
# Function to train the neural network
neuralnettrain <- function(ds, hl, lr, act, oldds) {
  traindata <- 1:380
  testdata <- (381:nrow(ds))
  neuralplot <- neuralnet(
    Prediction ~ .,
    data = ds[traindata,],
    hidden = hl,
    learningrate = lr,
    linear.output = TRUE,
    act.fct = act
  )
  #Unormalzing the data to be used on statistics
  unormdata <- Unnormalize(ds,oldds)
  predictions <- predict(neuralplot, newdata = unormdata[testdata,])[, 1]
  testing <- unormdata$Prediction[testdata]
  plot(neuralplot)
  return(list(
    neuralplot = neuralplot,
    SMAPE = smape(testing, predictions),
    MAPE = mape(testing, predictions),
    MAE = mae(testing, predictions),
    RMSE = rmse(testing, predictions)
  ))
}
# hidden layers and learning rates
LR1 <- 0.01
LR2 <- 0.1
onehiddenlayer <- c(3) # 3 nodes as single hidden layer
twohiddenlayer <- c(3, 3) # 3 nodes for each hidden layer

set.seed(100)
dataset <-
  read_excel("C:/Users/ASUS/Downloads/uow_consumption.xlsx")
names(dataset) <- c("Date", "18:00", "19:00", "20:00")
rowcount <- nrow(dataset)

ggplot(dataset, aes(x = Date, y = `20:00`)) +
  geom_point(color = "darkorange") +
  geom_smooth(color = "blue", method = "loess") +
  xlab("Date") +
  ylab("20th Hour Consumption")

#Input data-frames with relevant day gaps
T1 <- cbind(dataset$`20:00`[1:(rowcount - 1)],
            dataset$`20:00`[2:rowcount])%>%
  as.data.frame() # 20th hour and one day gap

T2 <- cbind(dataset$`20:00`[1:(rowcount - 2)],
            dataset$`20:00`[2:(rowcount - 1)],
            dataset$`20:00`[3:rowcount])%>%
  as.data.frame() # 20th hour and two day gap

T3 <- cbind(dataset$`20:00`[1:(rowcount - 3)],
            dataset$`20:00`[2:(rowcount - 2)],
            dataset$`20:00`[3:(rowcount - 1)],
            dataset$`20:00`[4:rowcount])%>%
  as.data.frame() # 20th hour and three day gap

T4 <- cbind(
  dataset$`20:00`[1:(rowcount - 4)],
  dataset$`20:00`[2:(rowcount - 3)],
  dataset$`20:00`[3:(rowcount - 2)],
  dataset$`20:00`[4:(rowcount - 1)],
  dataset$`20:00`[5:rowcount]
) %>%
  as.data.frame() # 20th hour and four day gap

T1T7 <- cbind(dataset$`20:00`[1:(rowcount - 7)],
              dataset$`20:00`[7:(rowcount - 1)],
              dataset$`20:00`[8:rowcount]) %>%
  as.data.frame() # with week gap

T2T7 <- cbind(dataset$`20:00`[1:(rowcount - 7)],
              dataset$`20:00`[6:(rowcount - 2)],
              dataset$`20:00`[7:(rowcount - 1)],
              dataset$`20:00`[8:rowcount])%>%
  as.data.frame()

T3T7 <- cbind(
  dataset$`20:00`[1:(rowcount - 7)],
  dataset$`20:00`[5:(rowcount - 3)],
  dataset$`20:00`[6:(rowcount - 2)],
  dataset$`20:00`[7:(rowcount - 1)],
  dataset$`20:00`[8:rowcount]
) %>%
  as.data.frame()


T4T7 <- cbind(
  dataset$`20:00`[1:(rowcount - 7)],
  dataset$`20:00`[4:(rowcount - 4)],
  dataset$`20:00`[5:(rowcount - 3)],
  dataset$`20:00`[6:(rowcount - 2)],
  dataset$`20:00`[7:(rowcount - 1)],
  dataset$`20:00`[8:rowcount]
) %>%
  as.data.frame()

T1norm <- Normalize(T1)
T2norm <- Normalize(T2)
T3norm <- Normalize(T3)
T4norm <- Normalize(T4)
T1T7norm <- Normalize(T1T7)
T2T7norm <- Normalize(T2T7)
T3T7norm <- Normalize(T3T7)
T4T7norm <- Normalize(T4T7)

names(T1norm) <- c("T1", "Prediction")
names(T2norm) <- c("T2", "T1", "Prediction")
names(T3norm) <- c("T3", "T2", "T1", "Prediction")
names(T4norm) <- c("T4", "T3", "T2", "T1", "Prediction")
names(T1T7norm) <- c("T7", "T1", "Prediction")
names(T2T7norm) <- c("T7", "T2", "T1", "Prediction")
names(T3T7norm) <- c("T7", "T3", "T2", "T1", "Prediction")
names(T4T7norm) <- c("T7", "T4", "T3", "T2", "T1", "Prediction")

#applying input dataset to neuralnetfunction to plot and show statistical indices
neuralnettrain(T1norm, onehiddenlayer, LR1, "logistic", T1)[2:5]
neuralnettrain(T1norm, onehiddenlayer, LR1, "tanh", T1)[2:5]
neuralnettrain(T1norm, onehiddenlayer, LR2, "logistic", T1)[2:5]
neuralnettrain(T1norm, onehiddenlayer, LR2, "tanh", T1)[2:5]
neuralnettrain(T1norm, twohiddenlayer, LR1, "logistic", T1)[2:5]
neuralnettrain(T1norm, twohiddenlayer, LR1, "tanh", T1)[2:5]
neuralnettrain(T1norm, twohiddenlayer, LR2, "logistic", T1)[2:5]
neuralnettrain(T1norm, twohiddenlayer, LR2, "tanh", T1)[2:5]

neuralnettrain(T1T7norm, onehiddenlayer, LR1, "logistic", T1T7)[2:5]
neuralnettrain(T1T7norm, onehiddenlayer, LR1, "tanh", T1T7)[2:5]
neuralnettrain(T1T7norm, onehiddenlayer, LR2, "logistic", T1T7)[2:5]
neuralnettrain(T1T7norm, onehiddenlayer, LR2, "tanh", T1T7)[2:5]
neuralnettrain(T1T7norm, twohiddenlayer, LR1, "logistic", T1T7)[2:5]
neuralnettrain(T1T7norm, twohiddenlayer, LR1, "tanh", T1T7)[2:5]
neuralnettrain(T1T7norm, twohiddenlayer, LR2, "logistic", T1T7)[2:5]
neuralnettrain(T1T7norm, twohiddenlayer, LR2, "tanh", T1T7)[2:5]
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T2norm, onehiddenlayer, LR1, "logistic", T2)[2:5]
neuralnettrain(T2norm, onehiddenlayer, LR1, "tanh", T2)[2:5]
neuralnettrain(T2norm, onehiddenlayer, LR2, "logistic", T2)[2:5]
neuralnettrain(T2norm, onehiddenlayer, LR2, "tanh", T2)[2:5]
neuralnettrain(T2norm, twohiddenlayer, LR1, "logistic", T2)[2:5]
neuralnettrain(T2norm, twohiddenlayer, LR1, "tanh", T2)[2:5]
neuralnettrain(T2norm, twohiddenlayer, LR2, "logistic", T2)[2:5]
neuralnettrain(T2norm, twohiddenlayer, LR2, "tanh", T2)[2:5]

neuralnettrain(T2T7norm, onehiddenlayer, LR1, "logistic", T2T7)[2:5]
neuralnettrain(T2T7norm, onehiddenlayer, LR1, "tanh", T2T7)[2:5]
neuralnettrain(T2T7norm, onehiddenlayer, LR2, "logistic", T2T7)[2:5]
neuralnettrain(T2T7norm, onehiddenlayer, LR2, "tanh", T2T7)[2:5]
neuralnettrain(T2T7norm, twohiddenlayer, LR1, "logistic", T2T7)[2:5]
neuralnettrain(T2T7norm, twohiddenlayer, LR1, "tanh", T2T7)[2:5]
neuralnettrain(T2T7norm, twohiddenlayer, LR2, "logistic", T2T7)[2:5]
neuralnettrain(T2T7norm, twohiddenlayer, LR2, "tanh", T2T7)[2:5]
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T3norm, onehiddenlayer, LR1, "logistic", T3)[2:5]
neuralnettrain(T3norm, onehiddenlayer, LR1, "tanh", T3)[2:5]
neuralnettrain(T3norm, onehiddenlayer, LR2, "logistic", T3)[2:5]
neuralnettrain(T3norm, onehiddenlayer, LR2, "tanh", T3)[2:5]
neuralnettrain(T3norm, twohiddenlayer, LR1, "logistic", T3)[2:5]
neuralnettrain(T3norm, twohiddenlayer, LR1, "tanh", T3)[2:5]
neuralnettrain(T3norm, twohiddenlayer, LR2, "logistic", T3)[2:5]
neuralnettrain(T3norm, twohiddenlayer, LR2, "tanh", T3)[2:5]

neuralnettrain(T3T7norm, onehiddenlayer, LR1, "logistic", T3T7)[2:5]
neuralnettrain(T3T7norm, onehiddenlayer, LR1, "tanh", T3T7)[2:5]
neuralnettrain(T3T7norm, onehiddenlayer, LR2, "logistic", T3T7)[2:5]
neuralnettrain(T3T7norm, onehiddenlayer, LR2, "tanh", T3T7)[2:5]
neuralnettrain(T3T7norm, twohiddenlayer, LR1, "logistic", T3T7)[2:5]
neuralnettrain(T3T7norm, twohiddenlayer, LR1, "tanh", T3T7)[2:5]
neuralnettrain(T3T7norm, twohiddenlayer, LR2, "logistic", T3T7)[2:5]
neuralnettrain(T3T7norm, twohiddenlayer, LR2, "tanh", T3T7)[2:5]
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T4norm, onehiddenlayer, LR1, "logistic", T4)[2:5]
neuralnettrain(T4norm, onehiddenlayer, LR1, "tanh", T4)[2:5]
neuralnettrain(T4norm, onehiddenlayer, LR2, "logistic", T4)[2:5]
neuralnettrain(T4norm, onehiddenlayer, LR2, "tanh", T4)[2:5]
neuralnettrain(T4norm, twohiddenlayer, LR1, "logistic", T4)[2:5]
neuralnettrain(T4norm, twohiddenlayer, LR1, "tanh", T4)[2:5]
neuralnettrain(T4norm, twohiddenlayer, LR2, "logistic", T4)[2:5]
neuralnettrain(T4norm, twohiddenlayer, LR2, "tanh", T4)[2:5]

neuralnettrain(T4T7norm, onehiddenlayer, LR1, "logistic", T4T7)[2:5]
neuralnettrain(T4T7norm, onehiddenlayer, LR1, "tanh", T4T7)[2:5]
neuralnettrain(T4T7norm, onehiddenlayer, LR2, "logistic", T4T7)[2:5]
neuralnettrain(T4T7norm, onehiddenlayer, LR2, "tanh", T4T7)[2:5]
neuralnettrain(T4T7norm, twohiddenlayer, LR1, "logistic", T4T7)[2:5]
neuralnettrain(T4T7norm, twohiddenlayer, LR1, "tanh", T4T7)[2:5]
neuralnettrain(T4T7norm, twohiddenlayer, LR2, "logistic", T4T7)[2:5]
neuralnettrain(T4T7norm, twohiddenlayer, LR2, "tanh", T4T7)[2:5]
#----------------------------------------------------------------------------------------------------------------
# NARX Approach
#taking columns 18:00 and 19:00 as per the specification
T1narx <- cbind(dataset$`20:00`[1:(rowcount - 1)],
                dataset$`18:00`[1:(rowcount - 1)],
                dataset$`19:00`[1:(rowcount - 1)],
                dataset$`20:00`[2:rowcount]) %>%
  as.data.frame()

T2narx <- cbind(
  dataset$`20:00`[1:(rowcount - 2)],
  dataset$`20:00`[2:(rowcount - 1)],
  dataset$`18:00`[2:(rowcount - 1)],
  dataset$`19:00`[2:(rowcount - 1)],
  dataset$`20:00`[3:rowcount]
) %>%
  as.data.frame()

T3narx <- cbind(
  dataset$`20:00`[1:(rowcount - 3)],
  dataset$`20:00`[2:(rowcount - 2)],
  dataset$`20:00`[3:(rowcount - 1)],
  dataset$`18:00`[3:(rowcount - 1)],
  dataset$`19:00`[3:(rowcount - 1)],
  dataset$`20:00`[4:rowcount]
) %>%
  as.data.frame()

T4narx <- cbind(
  dataset$`20:00`[1:(rowcount - 4)],
  dataset$`20:00`[2:(rowcount - 3)],
  dataset$`20:00`[3:(rowcount - 2)],
  dataset$`20:00`[4:(rowcount - 1)],
  dataset$`18:00`[4:(rowcount - 1)],
  dataset$`19:00`[4:(rowcount - 1)],
  dataset$`20:00`[5:rowcount]
) %>%
  as.data.frame()

T1T7narx <- cbind(
  dataset$`20:00`[1:(rowcount - 7)],
  dataset$`20:00`[7:(rowcount - 1)],
  dataset$`18:00`[7:(rowcount - 1)],
  dataset$`19:00`[7:(rowcount - 1)],
  dataset$`20:00`[8:rowcount]
) %>%
  as.data.frame()

T2T7narx <- cbind(
  dataset$`20:00`[1:(rowcount - 7)],
  dataset$`20:00`[6:(rowcount - 2)],
  dataset$`20:00`[7:(rowcount - 1)],
  dataset$`18:00`[7:(rowcount - 1)],
  dataset$`19:00`[7:(rowcount - 1)],
  dataset$`20:00`[8:rowcount]
) %>%
  as.data.frame()

T3T7narx <- cbind(
  dataset$`20:00`[1:(rowcount - 7)],
  dataset$`20:00`[5:(rowcount - 3)],
  dataset$`20:00`[6:(rowcount - 2)],
  dataset$`20:00`[7:(rowcount - 1)],
  dataset$`18:00`[7:(rowcount - 1)],
  dataset$`19:00`[7:(rowcount - 1)],
  dataset$`20:00`[8:rowcount]
) %>%
  as.data.frame()

T4T7narx <- cbind(
  dataset$`20:00`[1:(rowcount - 7)],
  dataset$`20:00`[4:(rowcount - 4)],
  dataset$`20:00`[5:(rowcount - 3)],
  dataset$`20:00`[6:(rowcount - 2)],
  dataset$`20:00`[7:(rowcount - 1)],
  dataset$`18:00`[7:(rowcount - 1)],
  dataset$`19:00`[7:(rowcount - 1)],
  dataset$`20:00`[8:rowcount]
) %>%
  as.data.frame()

T1narxnorm <- Normalize(T1narx)
T2narxnorm <- Normalize(T2narx)
T3narxnorm <- Normalize(T3narx)
T4narxnorm <- Normalize(T4narx)
T1T7narxnorm <- Normalize(T1T7narx)
T2T7narxnorm <- Normalize(T2T7narx)
T3T7narxnorm <- Normalize(T3T7narx)
T4T7narxnorm <- Normalize(T4T7narx)

names(T1narxnorm) <- c("T1narx", "Eighteenth_Hour", "Nineteenth_Hour", "Prediction")
names(T2narxnorm) <- c("T2narx", "T1narx", "Eighteenth_Hour", "Nineteenth_Hour", "Prediction")
names(T3narxnorm) <-c("T3narx", "T2narx", "T1narx", "Eighteenth_Hour", "Nineteenth_Hour", "Prediction")
names(T4narxnorm) <-c("T4narx","T3narx","T2narx","T1narx","Eighteenth_Hour","Nineteenth_Hour","Prediction")
names(T1T7narxnorm) <-c("T7narx", "T1narx", "Eighteenth_Hour", "Nineteenth_Hour", "Prediction")
names(T2T7narxnorm) <-c("T7narx", "T2narx", "T1narx", "Eighteenth_Hour", "Nineteenth_Hour", "Prediction")
names(T3T7narxnorm) <-c("T7narx","T3narx","T2narx","T1narx","Eighteenth_Hour","Nineteenth_Hour","Prediction")
names(T4T7narxnorm) <-c("T7narx","T4narx","T3narx","T2narx","Eighteenth_Hour","Nineteenth_Hour","T1narx","Prediction")
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T1narxnorm, onehiddenlayer, LR1, "logistic", T1narx)[2:5]
neuralnettrain(T1narxnorm, onehiddenlayer, LR1, "tanh", T1narx)[2:5]
neuralnettrain(T1narxnorm, onehiddenlayer, LR2, "logistic", T1narx)[2:5]
neuralnettrain(T1narxnorm, onehiddenlayer, LR2, "tanh", T1narx)[2:5]
neuralnettrain(T1narxnorm, twohiddenlayer, LR1, "logistic", T1narx)[2:5]
neuralnettrain(T1narxnorm, twohiddenlayer, LR1, "tanh", T1narx)[2:5]
neuralnettrain(T1narxnorm, twohiddenlayer, LR2, "logistic", T1narx)[2:5]
neuralnettrain(T1narxnorm, twohiddenlayer, LR2, "tanh", T1narx)[2:5]

neuralnettrain(T1T7narxnorm, onehiddenlayer, LR1, "logistic", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, onehiddenlayer, LR1, "tanh", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, onehiddenlayer, LR2, "logistic", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, onehiddenlayer, LR2, "tanh", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, twohiddenlayer, LR1, "logistic", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, twohiddenlayer, LR1, "tanh", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, twohiddenlayer, LR2, "logistic", T1T7narx)[2:5]
neuralnettrain(T1T7narxnorm, twohiddenlayer, LR2, "tanh", T1T7narx)[2:5]
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T2narxnorm, onehiddenlayer, LR1, "logistic", T2narx)[2:5]
neuralnettrain(T2narxnorm, onehiddenlayer, LR1, "tanh", T2narx)[2:5]
neuralnettrain(T2narxnorm, onehiddenlayer, LR2, "logistic", T2narx)[2:5]
neuralnettrain(T2narxnorm, onehiddenlayer, LR2, "tanh", T2narx)[2:5]
neuralnettrain(T2narxnorm, twohiddenlayer, LR1, "logistic", T2narx)[2:5]
neuralnettrain(T2narxnorm, twohiddenlayer, LR1, "tanh", T2narx)[2:5]
neuralnettrain(T2narxnorm, twohiddenlayer, LR2, "logistic", T2narx)[2:5]
neuralnettrain(T2narxnorm, twohiddenlayer, LR2, "tanh", T2narx)[2:5]

neuralnettrain(T2T7narxnorm, onehiddenlayer, LR1, "logistic", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, onehiddenlayer, LR1, "tanh", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, onehiddenlayer, LR2, "logistic", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, onehiddenlayer, LR2, "tanh", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, twohiddenlayer, LR1, "logistic", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, twohiddenlayer, LR1, "tanh", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, twohiddenlayer, LR2, "logistic", T2T7narx)[2:5]
neuralnettrain(T2T7narxnorm, twohiddenlayer, LR2, "tanh", T2T7narx)[2:5]
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T3narxnorm, onehiddenlayer, LR1, "logistic", T3narx)[2:5]
neuralnettrain(T3narxnorm, onehiddenlayer, LR1, "tanh", T3narx)[2:5]
neuralnettrain(T3narxnorm, onehiddenlayer, LR2, "logistic", T3narx)[2:5]
neuralnettrain(T3narxnorm, onehiddenlayer, LR2, "tanh", T3narx)[2:5]
neuralnettrain(T3narxnorm, twohiddenlayer, LR1, "logistic", T3narx)[2:5]
neuralnettrain(T3narxnorm, twohiddenlayer, LR1, "tanh", T3narx)[2:5]
neuralnettrain(T3narxnorm, twohiddenlayer, LR2, "logistic", T3narx)[2:5]
neuralnettrain(T3narxnorm, twohiddenlayer, LR2, "tanh", T3narx)[2:5]

neuralnettrain(T3T7narxnorm, onehiddenlayer, LR1, "logistic", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, onehiddenlayer, LR1, "tanh", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, onehiddenlayer, LR2, "logistic", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, onehiddenlayer, LR2, "tanh", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, twohiddenlayer, LR1, "logistic", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, twohiddenlayer, LR1, "tanh", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, twohiddenlayer, LR2, "logistic", T3T7narx)[2:5]
neuralnettrain(T3T7narxnorm, twohiddenlayer, LR2, "tanh", T3T7narx)[2:5]
#----------------------------------------------------------------------------------------------------------------
neuralnettrain(T4narxnorm, onehiddenlayer, LR1, "logistic", T4narx)[2:5]
neuralnettrain(T4narxnorm, onehiddenlayer, LR1, "tanh", T4narx)[2:5]
neuralnettrain(T4narxnorm, onehiddenlayer, LR2, "logistic", T4narx)[2:5]
neuralnettrain(T4narxnorm, onehiddenlayer, LR2, "tanh", T4narx)[2:5]
neuralnettrain(T4narxnorm, twohiddenlayer, LR1, "logistic", T4narx)[2:5]
neuralnettrain(T4narxnorm, twohiddenlayer, LR1, "tanh", T4narx)[2:5]
neuralnettrain(T4narxnorm, twohiddenlayer, LR2, "logistic", T4narx)[2:5]
neuralnettrain(T4narxnorm, twohiddenlayer, LR2, "tanh", T4narx)[2:5]

neuralnettrain(T4T7narxnorm, onehiddenlayer, LR1, "logistic", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, onehiddenlayer, LR1, "tanh", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, onehiddenlayer, LR2, "logistic", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, onehiddenlayer, LR2, "tanh", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, twohiddenlayer, LR1, "logistic", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, twohiddenlayer, LR1, "tanh", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, twohiddenlayer, LR2, "logistic", T4T7narx)[2:5]
neuralnettrain(T4T7narxnorm, twohiddenlayer, LR2, "tanh", T4T7narx)[2:5]