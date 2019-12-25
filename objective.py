import numpy as np 

def objective(z,*data):
    """ This is the objective function F outlined in the paper to minimize under the bounds, [0,0,0] and [20,1,1], and the constraint outlined in the constraint.py"""
    b,v1,v2,v3 = data
    
    a = z[0]
    ah = z[1]
    s = z[2]
    
    f = ((2*a*b**2)/(2*ah-s**2)+ah/a -v1)**2 + ((2*ah*(2*ah+s**2))/((2*a)**2)+((2*a)**2*b**4)/((2*ah-s**2)*(2*ah-2*s**2))+2*b**2-v2)**2 + ((2*ah*(2*ah*+s**2)*(2*ah*2*+s**2))/(8*a**3)+(8*a**3*b**6)/((2*ah-s**2)*(2*ah-2*s**2)*(2*ah-3*s**2))+3*b**2*ah/a+(6*a*b**4)/(2*ah - s**2)-v3)**2    
    return f