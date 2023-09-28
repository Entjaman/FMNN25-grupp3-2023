# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:25:32 2023

@author: dumi
Task 9(last 2 points)
• DFP rank-2 update
• BFGS rank-2 update


Task 12 : The matrix H(k) of the BFGS method should approximate G(x(k))−1, where G is
the Hessian of the problem. Study the quality of the approximation with growing k.


"""
import numpy as np
from methods import OptimizationMethod
from scipy.optimize import line_search


class QuasiNewtonMethod(OptimizationMethod):
    
    def __init__(self, opt_problem, tol = 1e-2, maxIters = 1000):
    

        self.opt_problem = opt_problem
        self.maxIters = maxIters
        self.tol = tol
 

    def minimize_bfgs(self, x_k):
        """
        The minimization iteration. Not sure if there should be some extra conditions for positive def
        for DFP(since BFGS preserves the pos def) or some other stopping condition?
        
        """
       
   
        H_k = np.eye(len(x_k), dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)
        print("s_k", s_k)
        i = 0
        # check the value of the gradient
        while i < self.maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > self.tol  :
           # line search to find alfa_k
           print("g_k", g_k)
           #### line search method ########
           alfa_k_scipy = line_search(self.opt_problem.function_value, self.opt_problem.gradient_value, x_k, s_k)[0]
           alfa_k = self.line_search_wolfe(x_k, s_k)
           print("my alfa",alfa_k)
           print("scipy alfa", alfa_k_scipy)
          # alfa_k=0.5
           # calculating delta should be a nx1 vector
           delta_k =alfa_k * s_k
           #print("delta_k",delta_k, delta_k.shape)
           # saving the gradient of the current x_k
           g_k = np.array(self.opt_problem.gradient_value(x_k))
           # updating the x_(k+1)
           x_k = x_k + delta_k
           # saving the gradient of the updated x_k
           g_updated = np.array(self.opt_problem.gradient_value(x_k))
           # calculating gamma
           gamma_k =  g_updated - g_k;
           #print("gamma_k", gamma_k)
           # temporary variables for better reading of the code i guess?
           # NxN matrix/scalar
           deltaTgamma =  np.dot(np.transpose(delta_k), gamma_k)
           # scalar /NxN matrix
           deltadeltaT = np.dot(delta_k, np.transpose(delta_k))

           H_k = H_k + (1 + np.dot(np.dot(np.transpose(gamma_k), H_k),gamma_k) / deltaTgamma) * (deltadeltaT/deltaTgamma) - np.dot(np.dot(delta_k , np.transpose(gamma_k)),H_k) + np.dot(np.dot(H_k ,gamma_k) , np.transpose(delta_k)) /deltaTgamma
           s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))
          # print(H_k)
           print("x_k", x_k)

           i= i+1
        print("Number of iterations BFGS:", i-1)
        print("x_k", x_k)
        return x_k
    
    
    def minimize_DFP (self, x_k):
        H_k = np.eye(len(x_k), dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)
        print("s_k", s_k)
        i = 0
        # check the value of the gradient
        while i < self.maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > self.tol  :
           # line search to find alfa_k
           print("g_k", g_k)
           #### line search method ########
           #alfa_k = line_search(self.opt_problem.function_value, self.opt_problem.gradient_value, x_k, s_k)[0]
           alfa_k = self.line_search_wolfe(x_k, s_k)

          # alfa_k=0.5
           # calculating delta should be a nx1 vector
           delta_k =alfa_k * s_k
           #print("delta_k",delta_k, delta_k.shape)
           # saving the gradient of the current x_k
           g_k = np.array(self.opt_problem.gradient_value(x_k))
           # updating the x_(k+1)
           x_k = x_k + delta_k
           # saving the gradient of the updated x_k
           g_updated = np.array(self.opt_problem.gradient_value(x_k))
           # calculating gamma
           gamma_k =  g_updated - g_k;
          # print("gamma_k", gamma_k)
           # temporary variables for better reading of the code i guess?
           # NxN matrix/scalar
           deltaTgamma =  np.dot(np.transpose(delta_k), gamma_k)
           
           # scalar /NxN matrix
           deltadeltaT = np.dot(delta_k, np.transpose(delta_k))
           
           
           H_k = H_k + deltadeltaT / deltaTgamma \
                        - np.dot(np.dot(np.dot(H_k, gamma_k), np.transpose(gamma_k)),H_k)\
                        / np.dot(np.dot(np.transpose(gamma_k),H_k),gamma_k) 
           
           i= i+1
           s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))
        print("Number of iterations DFS:", i-1)
        print("x_k", x_k)
        return x_k
        