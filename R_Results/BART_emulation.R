library(tgp)
library(scoringRules)
library(BART)
obs <- c(25, 24, 30, 31)
obs_dict  <- list()
obs_dict[[31]] <- 0
obs_dict[[24]] <- 1
obs_dict[[25]] <- 2
obs_dict[[30]] <- 3


folds <- 10
train_vec <- c(200, 250, 300, 350)
train_vec <- rev(train_vec)
final_list <- list()
for(n in train_vec) {
  ntrain <- n
  result_list <- list()
  for(o in obs) {
    rmse <- rep(NA, folds)
    crps <- rep(NA, folds)
    cov_vec <- rep(NA, folds)
    for(i in 1:folds) {
      print(n)
      print(o)
      print(i)
      X <- read.csv("au_design.txt")
      Y <- read.csv("au_y.txt")[,obs_dict[[o]]]
      
      randsamp <- sample(nrow(X))
      Xtot <- X[randsamp,]
      Ytot <- Y[randsamp]
      
      
      X <- as.matrix(Xtot[1:ntrain, ])
      Y <- Ytot[1:ntrain]
      
      
      Xtest <- as.matrix(Xtot[(ntrain  + 1):nrow(Xtot), ])
      Ytest <- Ytot[(ntrain + 1):nrow(Xtot)]
      
      
      X_sd = apply(X, 2, sd)
      X_mean = apply(X, 2, mean)
      
      Ysd <- sd(Y)
      Ym <- mean(Y)
      for(k in 1:ncol(Xtot)) {
        X[, k] = (X[, k] - X_mean[k])/X_sd[k]
        Xtest[, k] = (Xtest[, k] - X_mean[k])/X_sd[k]
      }
      #Xtot = (Xtot - X_mean)/X_sd
      Y <- (Y -Ym)/Ysd
      Ytest <- (Ytest -Ym)/Ysd
      
      
      what <- wbart(X, Y, Xtest, nskip = 20000, ndpost = 5000, printevery = 1000L)
      
      rmse[i] <- mean((what$yhat.test.mean - Ytest)^2)^(.5)

      
      #calculate crps
      zM <- what$yhat.test

      crps_vec <- rep(NA, ncol(zM))
      for(j in 1:ncol(zM)) {
        crps_vec[j] <- crps_sample(Ytest[j], zM[, j])
      }
      crps[i] <- mean(crps_vec)
      lower <- apply(zM, 2, function(x) quantile(x, .025))
      upper <- apply(zM, 2, function(x) quantile(x, .975))
      #cov <- mean(Ytest < upper & Ytest > lower)
      
      cov_vec[i] <- mean(Ytest < upper & Ytest > lower, na.rm = TRUE)
      print(rmse)
      print(crps)
      print(cov_vec)
    }
    result_list[[as.character(o)]] <- list(crps = crps, rmse = rmse, cov = cov_vec)
  }
  final_list[[as.character(n)]] <- result_list
}

saveRDS(final_list, file = "final_list_bart.rds")


final_list <- readRDS("final_list_bart.rds")
train_vec <- c(200, 250, 300, 350)
obs <- c(25, 24, 30, 31)

n_vec <- as.character(train_vec)
obs_vec <- as.character(obs)
folds_vec<- seq(1:folds)
val_vec <- c("rmse",  "crps", "cov")
result_frame <- expand.grid(n_vec ,obs_vec ,folds_vec, val_vec)
colnames(result_frame) <- c("n", "Observable", "Fold", "Statistic")

val <- rep(NA, nrow(result_frame))
for(i in 1:nrow(result_frame)) {
  frame_fold <- result_frame[i, ]$Fold
  inner_vec <- final_list[[as.character(result_frame[i, ]$n)]][[as.character(result_frame[i, ]$Observable)]][[as.character(result_frame[i, ]$Statistic)]]
  val[i] <- inner_vec[frame_fold]
}



result_frame$Value <- val

write.csv(result_frame, file = "sim_result_bart.csv", row.names = FALSE)

