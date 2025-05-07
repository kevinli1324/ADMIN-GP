library(tgp)
library(scoringRules)
library(lhs)


ht1 <- function(X) {
  return(sin(1*X[, 1]) + cos(.5*X[, 2]))
}

ht2 <- function(X) {
  return(sin(.5*X[, 1]*X[, 2]))
}

ht3 <- function(X) {
  return(X[, 2]*cos(.75*X[, 1]))
}


draw_data <- function(ntrain, ntest = 300, D = 20, stride = 6) {
  n = ntrain + ntest
  X = randomLHS(n, D) - .5
  
  M1 = matrix(rnorm(n = (D)*2), nrow = 2, ncol = D)
  M1[, 1:stride] <- matrix(rnorm(mean = 3, sd = 1, n = (stride*2)), nrow = 2, ncol = stride)
  M2 = matrix(rnorm(n = (D)*2), nrow = 2, ncol = D)
  M2[, (stride +1):(2*stride)] <- matrix(rnorm(mean = 3, sd = 1,n = (stride*2)), nrow = 2, ncol = stride)
  M3 = matrix(rnorm(n = (D)*2), nrow = 2, ncol = D)
  M3[, (2*stride +1):(3*stride)] <- matrix(rnorm(mean = 3, sd = 1,n = (stride*2)), nrow = 2, ncol = stride)
  
  nX1 = t(M1 %*% t(X))
  nX1 = nX1/sd(as.vector(nX1))
  nX2 = t(M2 %*% t(X))
  nX2 = nX2/sd(as.vector(nX2))
  nX3 = t(M3 %*% t(X))
  nX3 = nX3/sd(as.vector(nX3))
  
  Y = .4*ht1(nX1) + .3*ht2(nX2) + .3*ht3(nX3)
  Y = Y + rnorm(n = n)*.15*sd(Y)
  
  Xtest = X[(ntrain +1):n, ]
  X = X[1:ntrain, ]
  
  Ytest = Y[(ntrain +1):n]
  Y = Y[1:ntrain]
  
  return(list(X = X, Xtest = Xtest, Y =Y, Ytest = Ytest))

}

folds <- 10
train_vec <- c(500, 400, 300)
d_vec <- c(20, 25, 30)
final_list <- list()
for(D in d_vec) {
  result_list <- list()
  for(n in train_vec) {
      ntrain <- n
      rmse <- rep(NA, folds)
      crps <- rep(NA, folds)
      cov_vec <- rep(NA, folds)
      for(i in 1:folds) {
        print(n)
        print(i)
        
        samp_list = draw_data(ntrain = ntrain, D = D, stride = D %/% 3)
        Ytest  = samp_list[["Ytest"]]  
        Y  = samp_list[["Y"]]  
        X = samp_list[["X"]]
        Xtest = samp_list[["Xtest"]]
        
        X_sd = apply(X, 2, sd)
        X_mean = apply(X, 2, mean)
        
        Ysd <- sd(Y)
        Ym <- mean(Y)
        for(k in 1:ncol(X)) {
          X[, k] = (X[, k] - X_mean[k])/X_sd[k]
          Xtest[, k] = (Xtest[, k] - X_mean[k])/X_sd[k]
        }
        #Xtot = (Xtot - X_mean)/X_sd
        Y <- (Y -Ym)/Ysd
        Ytest <- (Ytest -Ym)/Ysd
        
        
        out <- bgp(X= X, Z=Y,XX = Xtest, corr="sim", trace = TRUE)
        rmse[i] <- sqrt(mean((Ytest - out$ZZ.mean)^2))
        
        #calculate crps
        zM <- out$trace$preds$ZZ
        fac = out$ZZ.mean[1]/mean(zM[, 1])
        zM <- fac*zM
        
        crps_vec <- rep(NA, ncol(zM))
        for(j in 1:ncol(zM)) {
          crps_vec[j] <- crps_sample(Ytest[j], zM[, j])
        }
        crps[i] <- mean(crps_vec)
        lower <- apply(zM, 2, function(x) quantile(x, .025))[1]
        upper <- apply(zM, 2, function(x) quantile(x, .975))[1]
        #cov <- mean(Ytest < upper & Ytest > lower)
        
        cov_vec[i] <- mean(Ytest < upper & Ytest > lower, na.rm = TRUE)
      }
    print( list(crps = crps, rmse = rmse, cov = cov_vec))
    result_list[[as.character(n)]] <- list(crps = crps, rmse = rmse, cov = cov_vec)
  }
  save_str <- paste0("mix_dat_", as.character(D), ".rds")
  saveRDS(result_list, file = save_str)
  final_list[[as.character(D)]] <- result_list
}

saveRDS(final_list, file = "mix_dat.rds")

list_20 <- readRDS("mix_dat_20.rds")
list_25 <- readRDS("mix_dat_25.rds")

final_list <- list()
final_list[[20]] <- list_20
final_list[[25]] <- list_25

folds <- 10
n_vec <- as.character(c(300, 400, 500))
d_vec <- c(20, 25)
folds_vec<- seq(1:folds)
val_vec <- c("rmse",  "crps", "cov")
result_frame <- expand.grid(n_vec ,d_vec ,folds_vec, val_vec)
colnames(result_frame) <- c("n", "Dimension", "Fold", "Statistic")

val <- rep(NA, nrow(result_frame))
for(i in 1:nrow(result_frame)) {
  frame_fold <- result_frame[i, ]$Fold
  inner_vec <- final_list[[result_frame[i, ]$Dimension]][[as.character(result_frame[i, ]$n)]][[as.character(result_frame[i, ]$Statistic)]]
  val[i] <- inner_vec[frame_fold]
}

result_frame$val <- val

write.csv(result_frame, file = "sim_sim.csv", row.names = FALSE)





