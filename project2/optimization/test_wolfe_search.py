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
import matplotlib.pyplot as plt

def obj_func(x):
    return (x[0])**2+(x[1])**2
def obj_grad(x):
    return [2*x[0], 2*x[1]]
def task12(inverse_hessians, true_inv_hessian):
    """
    Calculates the mean squared error of the inverse hessian approximations
    as compared to the "true" inverse hessian and plots it against the 
    number of iterations
    
    hessians : list of the hessians as returned by the minimize_bfgs()
    true_hessian : the hessian computed by inverting the approximation 
    calculated in Newton class. Can be done with np.linalg.inv()
    
    """
    mse = []
    for i in range(len(inverse_hessians)):
        mse.append( np.mean((inverse_hessians[i] - (true_inv_hessian))**2))
    print(mse)
    plt.plot(mse)
    plt.ylabel('mse')
    plt.xlabel('k')
    plt.show()
    
    


def main():

    # i think this should not be user defined, it should be a paramter that cannot be chnaged easily
    xk=np.linspace(0,1,11)    
    problem = OptimizationProblem(cp.chebyquad,cp.gradchebyquad)

    test_bfgs = quasi.QuasiNewtonMethod(problem)  
    x_k, hessians = test_bfgs.minimize_bfgs(xk, 1e-2,100, save_H= True)
    print("BFGS", x_k)
   # print(hessians)
    task12(hessians, np.linalg.inv(test_bfgs.hessian_aprox(xk, 1e-6)))
    # true_hessian = test_bfgs.hessian_aprox(xk, 1e-6)
    # mse = ((hessians - true_hessian)**2).mean()
    # print("MSE_shpe", len(hessians))
    #print(true_hessian)

   
    x_dfp = test_bfgs.minimize_DFP(xk, maxIters = 100)
    print("DFP", x_dfp)
    x_so = so.fmin_bfgs(cp.chebyquad, xk)
    print("BFGS scipy", x_so)
    print("cheby value for xk",cp.chebyquad(x_dfp))
    
    
if __name__ == "__main__":


    main()
        
