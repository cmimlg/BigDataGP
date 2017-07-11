library(laGP)
setwd("/home/admin123/BigDataIEEE/Review_Comments/airline_delay")
fp = "jan_feb_pp_data.csv"
df = read.csv(fp)
req.cols = c("DEP_DELAY", "TAXI_OUT", "TAXI_IN", "CRS_ELAPSED_TIME", "ARR_DELAY")
df = df[req.cols]
split.prop = 0.7
trng.set.size = split.prop * nrow(df)
trng.set.ind = sample(nrow(df), trng.set.size)
df.trng = df[trng.set.ind, ]
df.test = df[-trng.set.ind, ]
X = df.trng[, 1:4]
X = as.matrix(X)
Y = df.trng[, 5]
Y = as.matrix(Y)
Xt = df.test[, 1:4]
Xt = as.matrix(Xt)
Yt = df.test[, 5]
Yt = as.matrix(Yt)

batch.size = 15000
test.start = 1
test.end = test.start + batch.size
num.batches = ceiling(nrow(df.test)/batch.size)
Yp = vector()
test.set.start = 1
test.set.end = batch.size 
start.time = Sys.time()
for (i in seq(1:num.batches)) {
  cat("Processing batch " , i, "\n")
  Xtb = Xt[test.set.start:test.set.end,]
  out <- laGP(Xtb, 6,50, X, Y,method = "mspe")
  Yp = c(Yp, out$mean)
  test.set.start = test.set.end + 1
  if (test.set.end + batch.size > nrow(df.test)){
    test.set.end = nrow(df.test)
  }
  else {
    test.set.end = test.set.end + batch.size
  }
  
}
end.time = Sys.time()
total.time = end.time - start.time
cat("Total time for prediction is ", total.time, "\n")
err = Yt - Yp
se = err * err
rmse = sqrt(sum(se)/nrow(df.test))
cat("RMSE for the delay dataset is : " , rmse, "\n")

