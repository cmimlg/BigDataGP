import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from math import ceil, floor, exp, sqrt, log
import sys, traceback
import GPy
from GPy.kern import *
import random
import climin
import sys
import time
import os
import matplotlib.pyplot as plt
import logging
import csv
from memory_profiler import memory_usage
import statsmodels.api as sm


model_dict = dict()
model_estimates = None
N = 0
# create logger for the application
logger = logging.getLogger('Airline Delay Logger')

ch = logging.StreamHandler()





# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def init_model_estimates(num_test_pts):
    global model_estimates
    model_estimates = { i :list() for i in range(num_test_pts)}
    return


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


def do_sampling(K, delta, model_2, N_prop = 1.0):
    ERR_THRESHOLD = 8.75
    random.seed(42)
    t0 = time.time()
    #logger.debug("Reading data...")
    test_prop = 0.3
    X,Y, Xt, Yt = read_data(test_prop, N_prop)
    t1 = time.time()
    logger.debug("Data read took " + str(t1-t0) + " secs")
    logger.debug("Starting sampling...")
    Ntrng = X.shape[0]
    Xcs = Ycs = None
    if model_2:
        sample_size = int(ceil((Ntrng ** delta) * (1/0.5*(ERR_THRESHOLD ** 0.1))))
    else:
        sample_size = int(ceil(Ntrng ** delta))

    logger.debug ("sample size is " + str(sample_size))
    samples = dict()

    for i in range(K):
        sample_rows = random.sample(range(Ntrng), sample_size)
        Xs = X[sample_rows, ]
        Ys = Y[sample_rows, ]
        Ys = np.reshape(Ys, (len(Ys),1))
        samples[i] = (Xs,Ys)

        
    return samples, Xt, Yt
    



def sampling_gp(K = 30, delta = 0.56, model_2 = False, N_prop = 1.0):
    # Need to sample row-wise (including all attributes), so slightly different sampling
    # procedure.
    global model_dict

    logger.debug("Starting analysis...")

    samples, Xt, Yt = do_sampling(K, delta, model_2, N_prop)
 
    logger.debug("Calculating models...")
    t0 = time.time()
    num_test_pts = Xt.shape[0]
    init_model_estimates(num_test_pts)
    start_time = time.time()

    for i in range(K):


        try:
            sx = samples[i][0]
            sy = samples[i][1]

            k = RBF(input_dim = 4, ARD = True) + Linear(input_dim = 4)+\
                White(input_dim = 1, active_dims =[0])
            m  = GPy.models.GPRegression(sx, sy, k)
            m.optimize()

            model_dict[i] = m
            predict_model_estimates(Xt, Yt, i)
            #predict_POE_estimates(Xt, Yt, i)
            logger.debug("Completed K = " + str(i))
            
        except:
                logger.debug("Had an exception !")
                logger.debug ('-'*60)
                traceback.print_exc(file=sys.stdout)
                logger.debug ('-'*60)
    
    end_time = time.time()
    dt = (end_time - start_time)/60
    logger.debug("Time for opt + pred " + str(dt) + "minutes")
    logger.debug("Calculating errors")
    rmse, err = compute_errors(Yt)
 

    return rmse, err

def predict_model_estimates(Xt, Yt, K):
    global model_estimates, model_dict
    Z = zip(Xt,Yt)
    model = model_dict[K]
    Yp = model.predict(Xt)[0]
    Yp = Yp.ravel().tolist()

    for i in range(Xt.shape[0]):
        model_estimates[i].append(Yp[i])
    return

def predict_POE_estimates(Xt, Yt, K):
    global model_estimates, model_dict
    Z = zip(Xt,Yt)
    model = model_dict[K]
    Yp = model.predict(Xt)[0]
    Yp = Yp.ravel().tolist()
    varp = model.predict(Xt)[1]
    varp = varp.ravel().tolist()

    for i in range(Xt.shape[0]):
        model_estimates[i].append((Yp[i], varp[i]))
    return

def compute_POE_preds():
    global model_estimates
    Yhat = list()
    

    for m in model_estimates:
        est_tuples = model_estimates[m]
        point_precs = [1.0/est_tuples[i][1] for i in range(len(est_tuples))]
        nc = sum(point_precs)
        wts = [p/nc for p in point_precs]

        #calculate the weighted estimate for this point
        wt_ests = [wts[i]*est_tuples[i][0] for i in range(len(est_tuples))]
        point_est = sum(wt_ests)
        Yhat.append(point_est)

    return Yhat

def compute_avg_preds():
    global model_estimates
    Yhat = list()
    for m in model_estimates:
        pt_est = np.mean(model_estimates[m])
        Yhat.append(pt_est)        
    return Yhat


def compute_errors(Yt):
    num_test_pts = Yt.shape[0]
    #Yhat = compute_POE_preds()
    Yhat = compute_avg_preds()
    err = [(Yt[i] - Yhat[i]) for i in range(num_test_pts)]
    se = [(Yt[i] - Yhat[i])**2 for i in range(num_test_pts)]
    sse = sum(se)
    rmse = sqrt(sse/float(num_test_pts))

    return rmse, err




def K_analysis():
    global model_dict
    Ks = [ i for i in range(10, 100, 25)]
    rmses = list()
    for k in Ks:
        model_dict.clear()
        e = sampling_gp(k)
        rmses.append(e)
    return rmses

def higher_dim_SVGP_analysis(inducing_pts = 54):
    random.seed(1354)

    logger.debug("Starting analysis...")
    X,Y, Xt, Yt = read_data()
    Ntrng = X.shape[0]
    

    Xr = np.reshape(X, (X.shape[0], 4))
    Yr = np.reshape(Y, (Y.shape[0], 1))
    
    sample_rows = random.sample(range(Ntrng), inducing_pts)
    the_Z = Xr[sample_rows, ]
    the_Z = np.reshape(the_Z, (the_Z.shape[0], 4))
    logger.debug("Starting SVIGP...")
    rmse = None
    try:
        t0 = time.time()
        k = RBF(input_dim = 4, ARD = True) + Linear(input_dim = 4) + \
            White(input_dim = 1, active_dims = [0])


        m  = GPy.core.SVGP(Xr, Yr, the_Z, k,\
                           GPy.likelihoods.Gaussian(), batchsize = 5000)
        m.optimize()
        
        logger.debug(m)

        
        Yp = m.predict(Xt)[0]
        Yp = Yp.ravel().tolist()
        # Compute avg error for each point
 
        Ntest = Yt.shape[0]
        sse = sum([abs(Yp[i] - Yt[i])**2 for i in range(Ntest)])
        rmse = sqrt(sse/float(Ntest))
        t1 = time.time()
        dt = t1 -t0
        logger.debug("Prediction + training took " + str(dt) + " seconds")
    except:
            logger.debug("Had an exception !")
            logger.debug ('-'*60)
            traceback.print_exc(file=sys.stdout)
            logger.debug ('-'*60)


    logger.debug("Done!")
    
    return rmse

def higher_dim_Sparse_GP_analysis(ni = 54):
   
    random.seed(1354)
    
    logger.debug("Starting analysis...")
    X,Y, Xt, Yt = read_data()
    Ntrng = X.shape[0]

    Xr = np.reshape(X, (X.shape[0], 4))
    Yr = np.reshape(Y, (Y.shape[0], 1))
   
    logger.debug("Starting Sparse GP...")
    logger.debug("Number of inducing points = " + str(ni))
    rmse = None
    try:
        
        t0 = time.time()
        k = RBF(input_dim = 4, ARD = True) + Linear(input_dim = 4) +\
                White(input_dim = 1, active_dims = [0])
        m  = GPy.models.SparseGPRegression(Xr, Yr, k, num_inducing = ni)
        m.optimize()
        logger.debug(m)
        Yp = m.predict(Xt)[0]
        Yp = Yp.ravel().tolist()
        # Compute avg error for each point
        Ntest = Yt.shape[0]
        sse = sum([abs(Yp[i] - Yt[i])**2 for i in range(Ntest)])
        rmse = sqrt(sse/float(Ntest))
        t1 = time.time()
        dt = t1 - t0
        logger.debug("Prediction + training took " + str(dt) + " seconds")
        
    except:
            logger.debug("Had an exception !")
            logger.debug ('-'*60)
            traceback.print_exc(file=sys.stdout)
            logger.debug ('-'*60)

 

    logger.debug("Done!")

    return rmse

def exp_delta_versus_N_eps_fixed(err_threshold = 10.65):
    global N
    N_prop = np.linspace(0.1, 1, 10)
    delta_inc = 0.01 
    result = []
    random.seed(1254)
    # Starting value of delta is that which is needed for 10 samples
    # for that N
    for frac in N_prop:
        
        if frac == 0.1:
            delta = log(10)/log(8.38e4) # corresponds to smallest Ntrng
        else:
            delta = log(10)/log(Ntrng) # Use this for the others
        
        rmse, err = sampling_gp(30, delta, False, frac)
        Ntrng = frac * N * 0.7 # 70 % training
        logger.info ("Running N = " + str(Ntrng))
        gap = rmse - err_threshold
        while gap > 0.1:
            logger.info("RMSE experiment : " + str(rmse))
            delta = delta + delta_inc
            logger.info("delta increased to " + str(delta))
            rmse, err = sampling_gp(30, delta, False, frac)
            gap = rmse - err_threshold
        logger.info("Final RMSE experiment : " + str(rmse))    
        result.append([Ntrng, delta, rmse])

    os.chdir("/home/admin123/Big_Data_Paper_Code_Data/TKDE_Review_Exp/"\
             "AirlineDelay/jan_feb_2016_data")
    with open('airline_delay_exp_delta_versus_N_eps_fixed.csv', 'w') as fp:
        cw = csv.writer(fp, delimiter=',')
        cw.writerows(result)

    #do the plots
    Nexp = len(result)
    llN = [log(log(result[i][0])) for i in range(Nexp)]
    ds = [result[i][1] for i in range(Nexp)]
    plt.scatter(llN, ds, color = "blue")
    plt.title("log(log(N)) Versus $\delta$, $\epsilon$ = " +\
                  str(err_threshold) + " , K = 30")
    plt.ylabel("$\delta$")
    plt.xlabel("log(log(N))")
    plt.grid()
    plt.show()
    return 

def exp_delta_versus_eps_fixed_N(N_prop = 1.7e-3):
    random.seed(1254)

    deltas = np.linspace(0.16667, 0.8, 10)
    result = []    
    for d in deltas:
        logger.info("Running experiment for delta = %f" %d)
        rmse, err = sampling_gp(30, d, False, N_prop)
        result.append([d, rmse])

    os.chdir("/home/admin123/Big_Data_Paper_Code_Data/TKDE_Review_Exp/"\
             "AirlineDelay/jan_feb_2016_data")
    with open('airline_delay_delta_versus_eps_fixed_N.csv', 'w') as fp:
        cw = csv.writer(fp, delimiter=',')
        cw.writerows(result)

    #do the plots
    N_for_exp = int(838362*N_prop)
    Nexp = len(result)
    ds = [result[i][0] for i in range(Nexp)]
    eps = [result[i][1] for i in range(Nexp)]
    plt.scatter(ds, eps, color = "blue")
    plt.title("$\delta$ Versus $\epsilon$, N = " + str(N_for_exp) + " , K = 30")
    plt.ylabel("$\epsilon$")
    plt.xlabel("$\delta$")
    plt.grid()
    plt.show()
    return 

def exp_K_versus_eps_fixed_delta(N_prop = 0.6, delta = 0.30):
    random.seed(1254)
    K = np.linspace(1,100, 30)
    result = []

    for k in K:
        logger.info("Running experiment for K = %d" %k)
        rmse, err = sampling_gp(int(k), delta, False, N_prop)
        result.append([int(k), rmse])
        #do the plots
    
    os.chdir("/home/admin123/Big_Data_Paper_Code_Data/TKDE_Review_Exp/"\
             "AirlineDelay/jan_feb_2016_data")
    with open('airline_delay_exp_K_versus_eps_fixed_delta.csv', 'w') as fp:
        cw = csv.writer(fp, delimiter=',')
        cw.writerows(result)

    Nexp = len(result)
    N_for_exp = int(838362*N_prop)
    Ks = [result[i][0] for i in range(Nexp)]
    eps = [result[i][1] for i in range(Nexp)]
    plt.scatter(Ks, eps, color = "blue")
    plt.title("K versus $\epsilon$  N = " + str(N_for_exp) + ", $\delta = $" + str(delta))
    plt.ylabel("$\epsilon$")
    plt.xlabel("K")
    plt.grid()
    plt.show()
    return

def profile_mem_bagging():
    
    mem_usage =memory_usage((sampling_gp, (30, 0.3, False, 1.0)))
    max_mem = max(mem_usage)
    logger.info("Max memory for bagging GP was " + str(max_mem) + " MB")

    return max_mem

def profile_Sparse_GP():
    mem_usage =memory_usage((higher_dim_Sparse_GP_analysis))
    max_mem = max(mem_usage)
    logger.info("Max memory for Sparse GP was " + str(max_mem) + " MB")
    return max_mem

def profile_SVGP_GP():
    mem_usage =memory_usage((higher_dim_SVGP_analysis))
    max_mem = max(mem_usage)
    logger.info("Max memory for SVGP GP was " + str(max_mem) + " MB")
    return max_mem
    

def err_analysis():
    rmse, err = sampling_gp()
    err = np.array(err)
    pp = sm.ProbPlot(err, fit = True)
    pp.ppplot(line = '45')
    plt.grid()
    plt.show()
    plt.title("PP Plot for Airline Delay data")

    return    
    
