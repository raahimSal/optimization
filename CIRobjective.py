import scipy.stats as sc
import scipy.linalg as lin
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import csv


def CIRobjective(params,data):
    """Log-likelihood objective funciton for the CIR Process using the """ 
    dataF = data[1:]
    dataL = data[:len(data)-1]
    nobs = len(data)
    delta = 1/250
    alpha = params[0]
    mu = params[1]
    sigma = params[2]
    
    c = 2*alpha/(sigma**2*(1-np.exp(-alpha * delta)))
    q = 2 * alpha * mu/sigma**2-1
    u = c * np.exp(-alpha*delta)*dataL
    v = c*dataF
    nc = 2*u
    df = 2*q + 2
    s = 2*v
    
    gpdf = sc.ncx2.pdf(s,df,nc)
    ppdf = 2 *c*gpdf
    lnL = np.sum(-np.log(ppdf))
    return lnL
    