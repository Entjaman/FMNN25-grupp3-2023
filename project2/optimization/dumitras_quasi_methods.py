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
from methods import OptimizationProblem

from scipy.optimize import line_search


class QuasiNewtonMethod():
    
    def __init__(self, opt_problem, tol = 1e-2, maxIters = 1000):
    

        self.opt_problem = opt_problem
        self.maxIters = maxIters
        self.tol = tol
     
        
        
    def line_search_wolfe(self,xk, sk,c1=1e-20,c2=0.9, maxIter=50):
        
        """
        optimization_problem: for extracting the objective function and the gradient
        xk                  : the x values at iteration k
        sk                  : search direction = -H * gk at iteration k
        c1                  : arbitrary constant between (0,1)
        c2                  : arbitrary constant for the curvature condition between (c1,1)
        
        Implemented as per slide 21 in the course lecture
        """
        alfa_k_minus = 1 
        
        fx = np.array(self.opt_problem.function_value(xk))
        dfx = np.array(self.opt_problem.gradient_value(xk))
        
        def phi(alfa):
            return np.array(self.opt_problem.obj_function(xk + alfa*sk))
        def dphi(alfa):
            return np.array(self.opt_problem.gradient_value(xk + alfa*sk))
        iteration = 0
        # Armijo rule not fullfiled
        while phi(alfa_k_minus) > fx + c1*alfa_k_minus* np.dot(dfx.T, sk) and iteration<maxIter:
            alfa_k_minus = alfa_k_minus /2
            iteration = iteration + 1
        alfa_k_plus = alfa_k_minus 
       # print(iteration)
        iteration = 0
        while phi(alfa_k_plus) <= fx + c1*alfa_k_plus* np.dot(dfx.T, sk) and iteration<maxIter:
            alfa_k_plus = 2 * alfa_k_plus
            iteration = iteration + 1
       # print(iteration)
        # Curvature rule not fullfiled 
        iteration = 0
        while np.dot(dphi(alfa_k_minus).T, sk) < c2 * np.dot(dfx.T, sk) and iteration<maxIter :
            alfa_zero = (alfa_k_plus+alfa_k_minus)/2
            if phi(alfa_zero) <= fx + c1*alfa_zero* np.dot(dfx.T, sk):
                alfa_k_minus = alfa_zero
            else:
                alfa_k_plus = alfa_zero
            iteration = iteration + 1
        #print(iteration)
        return alfa_k_minus        
            
 

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
        print("Number of iterations:", i-1)
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
        print("Number of iterations:", i-1)
        print("x_k", x_k)
        return x_k
        