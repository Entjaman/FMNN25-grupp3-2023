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
    
    def __init__(self, opt_problem):
    

        self.opt_problem = opt_problem 

    def minimize_bfgs(self, x_k, tol= 1e-10, maxIters = 50, save_H = False  ):
        """
        The minimization iteration. Not sure if there should be some extra conditions for positive def
        for DFP(since BFGS preserves the pos def) or some other stopping condition?
        
        """
      
        n = len(x_k)
        H_k = np.eye(n, dtype= float)
        reg = 0.0001 * np.eye(n, dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)
        hessians = []

      #  print("s_k", s_k)
        i = 0
        # check the value of the gradient
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol  :
           # line search to find alfa_k
            try:
               np.linalg.cholesky(H_k)
            except:
               return False

        
          
          # print("g_k", g_k)
           #### line search method ########
         #  alfa_k = line_search(self.opt_problem.function_value, self.opt_problem.gradient_value, x_k, s_k)[0]
            alfa_k = self.line_search_wolfe(x_k, s_k)

            delta_k =alfa_k * s_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
            x_k = x_k + delta_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
            gamma_k =  g_updated - g_k;

           # bfgs update
            # scalar
            gTHg =np.dot(np.dot(np.transpose(gamma_k),H_k), gamma_k)
            dTg =  np.dot(np.transpose(delta_k),gamma_k)
            # nxn matrix 
            ddT =  np.dot(np.reshape(delta_k,(n,1)),np.reshape(delta_k, (1,n)))
            dgTH = np.dot(np.dot( np.reshape(delta_k, (n,1)), np.reshape(gamma_k, (1,n))), H_k)
            HgdT = np.dot(np.dot(H_k ,np.reshape(gamma_k, (n,1))) , np.reshape(delta_k, (1,n)))



            H_k = H_k + reg + ( 1 + gTHg / dTg) * (ddT / dTg) - (dgTH + HgdT) / dTg
            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))

            i= i+1
            if save_H == True:
                hessians = hessians.append(H_k)
                
        print("Number of iterations BFGS:", i-1)
        print("x_k", x_k)
        return x_k, hessians
    
    
    def minimize_DFP (self, x_k, tol= 1e-10, maxIters = 50 ):
        H_k = np.eye(len(x_k), dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)
        
        i = 0
        # check the value of the gradient
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol  :
            # line search to find alfa_k
            try:
              np.linalg.cholesky(H_k)
            except:
            #  print("Number of iterations DFP:", i-1)
              return False
           
           
          # print("g_k", g_k)
           #### line search method ########
           #alfa_k = line_search(self.opt_problem.function_value, self.opt_problem.gradient_value, x_k, s_k)[0]
            alfa_k = self.line_search_wolfe(x_k, s_k)

          # alfa_k=0.5
           # calculating delta should be a nx1 vector
            delta_k =alfa_k * s_k
           # saving the gradient of the current x_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
           # updating the x_(k+1)
            x_k = x_k + delta_k
           # saving the gradient of the updated x_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
           # calculating gamma
            gamma_k =  g_updated - g_k;
           # temporary variables for better reading of the code i guess?
           # NxN matrix/scalar
            dTg =  np.dot(np.transpose(delta_k), gamma_k)
           
           #  /NxN matrix
            ddT = np.dot(delta_k, np.transpose(delta_k))     
            HggTH = H_k @ gamma_k  @ np.transpose(gamma_k) * H_k
            gTHg = np.transpose(gamma_k) @ H_k @ gamma_k  
            H_k = H_k  + 0.0001 * np.eye(len(x_k), dtype= float) + ddT/ dTg  - HggTH / gTHg
            i= i+1
          # print("H_k", H_k)
            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))
        print("Number of iterations DFP:", i-1)
       # print("x_k", x_k)
        return x_k
        