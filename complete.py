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
    

def CIRestimation(model):
    
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
    
    #kkr = opt.basinhopping(CIRobjective,x0, minimizer_kwargs= {'args':model})
    
    return kkr

def objective(z,*data):
    b,v1,v2,v3 = data
    
    a = z[0]
    ah = z[1]
    s = z[2]
    
    f = ((2*a*b**2)/(2*ah-s**2)+ah/a -v1)**2 + ((2*ah*(2*ah+s**2))/((2*a)**2)+((2*a)**2*b**4)/((2*ah-s**2)*(2*ah-2*s**2))+2*b**2-v2)**2 + ((2*ah*(2*ah*+s**2)*(2*ah*2*+s**2))/(8*a**3)+(8*a**3*b**6)/((2*ah-s**2)*(2*ah-2*s**2)*(2*ah-3*s**2))+3*b**2*ah/a+(6*a*b**4)/(2*ah - s**2)-v3)**2    
    return f

def constraint(z):
    c = 3*z[2]**2-2*z[1]
    return 0 - c


## Estimation.py

VVIX = pd.read_csv('vvstoxx.csv', header=-1, usecols=[1])/100
a = 1
b = np.power(VVIX.min(),2)/4
Vplus = ((VVIX + np.sqrt(VVIX**2-4*a*b))/(2*a))**2
Vminus = ((VVIX -np.sqrt(VVIX**2-4*a*b))/(2*a))**2

model1 = Vplus
model2 = Vminus
res1 = CIRestimation(model1)
res2 = CIRestimation(model2)

V = Vplus + Vminus
V1 = np.mean(V)
V2 = np.mean(V**2)
V3 = np.mean(V**3)

aplus = res1.x[0]
ahplus = res1.x[0]*res1.x[1]
splus = res1.x[2]
initialplus = np.array([aplus,ahplus,splus])

aminus = res2.x[0]
ahminus = res2.x[0]*res2.x[1]
sminus = res2.x[2]
initialminus = np.array([aminus,ahminus,sminus])

lb = np.array([0,0,0])
ub = np.array([20,1,1])

#bounds = opt.Bounds(lb,ub)
bb = [(0,20),(0,1),(0,1)]

cons = opt.NonlinearConstraint(constraint, 0, +np.inf)

#cons =({'type': 'ineq', 'fun':constraint})
args=(b,V1,V2,V3)
#sol = opt.minimize(objective,initialminus, args = (b,V1,V2,V3), constraints = cons, options={'disp':True})
#solly = opt.differential_evolution(objective, bounds, args=args, constraints = cons, disp = True)
#sol = opt.shgo(objective, bb , args = args, constraints = cons)

## For this optimizer if youre planning on using any iter >150  make disp = False!!! 
sol = opt.basinhopping(objective, initialminus, niter = 1000, minimizer_kwargs={'bounds':bb,'constraints':cons, 'args':args},stepsize = 0.003, disp = True)