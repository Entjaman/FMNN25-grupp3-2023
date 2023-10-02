# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:10:38 2023
Testbench for the Wolfe search line method
@author: dumi
"""
import numpy as np

from scipy.optimize import line_search
import scipy.optimize as so
from optimization_problem import OptimizationProblem
from methods import OptimizationMethod
import chebyquad_problem as cp
import dumitras_quasi_methods as quasi

def obj_func(x):
    return (x[0])**2+(x[1])**2
def obj_grad(x):
    return [2*x[0], 2*x[1]]

def main():

    xk = np.array([1.34, 1.45,1,1.5,1,1,1,1.5])
    xk=np.linspace(0,1,11) #This is our initial guess. Should we add that as an input to our methods as well?
    sk = np.array([-1.00, -1.00])
   # alfa = line_search(cp.chebyquad, cp.gradchebyquad, xk, sk)[0]
   # print("scipy search",alfa)
    #print("my search",alfa)
    
    problem = OptimizationProblem(cp.chebyquad,cp.gradchebyquad)
  #  problem = OptimizationProblem(obj_func, obj_grad)
    #line_search_test = OptimizationMethod(problem)
    #alfa = line_search_test.line_search_wolfe(xk, sk)
    
    test_bfgs = quasi.QuasiNewtonMethod(problem)
   
    x_k = test_bfgs.minimize_bfgs(xk, 1e-2,100)
    print("BFGS", x_k)
   
    x_dfp = test_bfgs.minimize_DFP(xk)
    print("DFP", x_dfp)
    x_so = so.fmin_bfgs(cp.chebyquad, xk)
    print("BFGS scipy", x_so)
    print("cheby value for xk",cp.chebyquad(x_dfp))
if __name__ == "__main__":


    main()
        
