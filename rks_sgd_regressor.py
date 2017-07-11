from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import pandas as pd
import random
import numpy as np
import math
from math import ceil, floor, exp, sqrt, log

import time
import os
import logging

model_dict = dict()
model_estimates = None
N = 0
# create logger for the application
logger = logging.getLogger('Airline Delay RKS Logger')
ch = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


ch = logging.StreamHandler()
def read_data(test_set_prop = 0.3, N_prop = 1.0):
    global N
    X = Y = Xt = Yt = None
    os.chdir("/home/admin123/BigDataIEEE/Review_Comments/airline_delay")
             

    fp = "jan_feb_pp_data.csv"
    df = pd.read_csv(fp)
    #Variable selection is needed to reduce error for this dataset
    df = df[["DEP_DELAY", "TAXI_OUT", "TAXI_IN", "CRS_ELAPSED_TIME", "ARR_DELAY"]]
    
    N = len(df.index)
    N_exp = int(N*N_prop) - 1
    df = df.sample(frac = N_prop, random_state = 1254)
    test_set_size = int(ceil(N_exp * test_set_prop))
    test_rows = random.sample(df.index, test_set_size)
    df_trng = df.drop(test_rows)
    df_test = df.ix[test_rows]
    Ntrng = len(df_trng.index)
    X = df_trng.ix[:, 0:4].as_matrix()
    Y = df_trng.ix[:, 4].as_matrix()
    Y = Y.astype(np.float64)
    Xt = df_test.ix[:,0:4].as_matrix()
    Yt = df_test.ix[:,4].as_matrix()
    Yt = Yt.astype(np.float64)
    Ntrng = X.shape[0]
    Ntest = Xt.shape[0]

    logger.debug("Training set size is " + str(Ntrng))
    logger.debug("Test set size is " + str(Ntest))

    return X,Y, Xt, Yt

def do_RKS():
    X,Y, Xt, Yt = read_data()
    rbf_feature = RBFSampler(gamma=0.1, random_state=1)
    X_features = rbf_feature.fit_transform(X)
    Xt_features = rbf_feature.fit_transform(Xt)
    sgdreg = SGDRegressor(penalty='l2', alpha=0.15, n_iter=200)
    sgdreg.fit(X_features,Y)
    Yp = sgdreg.predict(Xt_features)
    err = Yt - Yp
    se = err * err
    Nt = Xt.shape[0]
    rmse = sqrt(sum(se)/Nt)
    logger.info("RMSE for airline delay dataset using RKS is " + str(rmse))
    return

    
    
