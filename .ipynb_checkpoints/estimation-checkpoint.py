## Estimation.py
import scipy.stats as sc
import scipy.linalg as lin
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import csv
import CIRobjective

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

bounds = opt.Bounds(lb,ub)
#bb = [(0,20),(0,1),(0,1)]

cons = opt.NonlinearConstraint(constraint, 0, +np.inf)

#cons =({'type': 'ineq', 'fun':constraint})
args=(b,V1,V2,V3)
#sol = opt.minimize(objective,initialminus, args = (b,V1,V2,V3), constraints = cons, options={'disp':True})
#solly = opt.differential_evolution(objective, bounds, args=args, constraints = cons, disp = True)
#sol = opt.shgo(objective, bb , args = args, constraints = cons)
sol = opt.basinhopping(objective, initialminus, niter = 1000, minimizer_kwargs={'bounds':bb,'constraints':cons, 'args':args},stepsize = 0.003, disp = True)