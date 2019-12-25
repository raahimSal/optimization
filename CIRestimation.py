import scipy.stats as sc
import scipy.linalg as lin
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import csv
import CIRobjective


def CIRestimation(model):
    """ This function minimizes the log-likelihood function outlined in the CIRObjective.py file"""  
    delta = 1/250

    nobs = len(model)
    x = model[:nobs-1]
    dx = -model.diff(periods = -1)
    dx = dx.dropna()
    dx = dx/x**0.5

    df1 = delta/x**0.5
    df2 = delta*x**0.5
    regressors = pd.concat([df1,df2], axis = 1)
    
    drift = np.linalg.lstsq(regressors,dx,rcond=None)[0] #lstsq is the OLS solver
    res = np.subtract(regressors@drift,dx)

    alpha = -np.float(drift[1])
    mu = -np.float(drift[0]/drift[1])
    sigma = np.float(np.sqrt(np.var(res)/delta))

    x0 = np.array([alpha,mu,sigma])
    
    """Here are the optimizer settings we can tweak. So far I am using a local minimizer
    with the Nelder-Mead method. The Global Optimization kit within SciPy requires bounds
    for which to find the global mimimum over, it achieves this by running a lot of local
    minimizers over the total space. To tweak """
    
    kkr = opt.minimize(CIRobjective,x0,args= model,method='Nelder-Mead')
    
    return kkr