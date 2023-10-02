import numpy as np
from methods import OptimizationMethod

class QuasiNewtonMethod(OptimizationMethod):
    
    def __init__(self, opt_problem):
    
        self.opt_problem = opt_problem 

    def minimize_good_broyden(self, x_k, tol= 1e-10, maxIters = 50):
        n = len(x_k)
        H_k = np.eye(n, dtype= float)
        reg = 0.0001 * np.eye(n, dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)

        i = 0
        # check the value of the gradient
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol:
            alfa_k = self.line_search_wolfe(x_k, s_k)
            delta_k =alfa_k * s_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
            x_k = x_k + delta_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
            gamma_k =  g_updated - g_k
            Hg = np.dot(H_k + reg, gamma_k)
            dTHg = np.dot(np.dot(delta_k.T, H_k+reg), gamma_k)
            dTH = np.dot(delta_k.T,H_k+reg)
            H_k = H_k + ((delta_k - Hg)/(dTHg))*dTH

            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))

            i= i+1

        print("Number of iterations Good Broyden:", i-1)
        return x_k
    

    def minimize_bad_broyden(self, x_k, tol= 1e-10, maxIters = 50):
        n = len(x_k)
        H_k = np.eye(n, dtype= float)
        reg = 0.0001 * np.eye(n, dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)

        i = 0
        # check the value of the gradient
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol:
            alfa_k = self.line_search_wolfe(x_k, s_k)
            delta_k =alfa_k * s_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
            x_k = x_k + delta_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
            gamma_k =  g_updated - g_k
            Hg = np.dot(H_k + reg, gamma_k)
            gTg = np.dot(gamma_k.T, gamma_k)
            
            H_k = H_k + ((delta_k - Hg)/(gTg))*gamma_k.T

            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))

            i= i+1

        print("Number of iterations Bad Broyden:", i-1)
        return x_k


    def minimize_symmetric_broyden(self, x_k, tol= 1e-10, maxIters = 50):
        n = len(x_k)
        H_k = np.eye(n, dtype= float)
        reg = 0.0001 * np.eye(n, dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)

        i = 0
        # check the value of the gradient
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol:
            alfa_k = self.line_search_wolfe(x_k, s_k)
            delta_k =alfa_k * s_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
            x_k = x_k + delta_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
            gamma_k =  g_updated - g_k
            
            u = delta_k - np.dot(H_k, gamma_k)
            a = 1/np.dot(u.T, gamma_k)
            uu = np.dot(u, u.T)

            H_k = H_k + reg+ np.dot(np.dot(a, u), u.T)

            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))

            i= i+1

        print("Number of iterations Bad Broyden:", i-1)
        return x_k

    def minimize_bfgs(self, x_k, tol= 1e-10, maxIters = 50, save_H = False  ):
        
        """
        x_k : initial values
        tol : the maximum tolerated value for the gradient
        maxIters : maximum number of iterations
        save_H   : saves the computed hessians if needed
        
        """
      
        n = len(x_k)
        H_k = np.eye(n, dtype= float)
        reg = 0.0001 * np.eye(n, dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)
        hessians = [H_k]
        i = 0
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol  :
            try:
               np.linalg.cholesky(H_k)
            except:
               return False
            alfa_k = self.line_search_wolfe(x_k, s_k)

            delta_k =alfa_k * s_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
            x_k = x_k + delta_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
            gamma_k =  g_updated - g_k

            gTHg =np.dot(np.dot(np.transpose(gamma_k),H_k), gamma_k)
            dTg =  np.dot(np.transpose(delta_k),gamma_k)
            ddT =  np.dot(np.reshape(delta_k,(n,1)),np.reshape(delta_k, (1,n)))
            dgTH = np.dot(np.dot( np.reshape(delta_k, (n,1)), np.reshape(gamma_k, (1,n))), H_k)
            HgdT = np.dot(np.dot(H_k ,np.reshape(gamma_k, (n,1))) , np.reshape(delta_k, (1,n)))


            H_k = H_k + reg + ( 1 + gTHg / dTg) * (ddT / dTg) - (dgTH + HgdT) / dTg
            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))

            i= i+1
            if save_H == True:
                hessians.append(H_k)
                
        print("Number of iterations BFGS:", i-1)
        return x_k, hessians
    
    
    def minimize_DFP (self, x_k, tol= 1e-10, maxIters = 50 ):
       
        n = len(x_k)
        H_k = np.eye(n, dtype= float)
        reg = 0.0001 * np.eye(n, dtype= float)
        g_k = self.opt_problem.gradient_value(x_k)
        s_k = np.dot(-H_k,g_k)
        i = 0
        
        while i < maxIters and  np.linalg.norm(self.opt_problem.gradient_value(x_k)) > tol  :
            np.linalg.cholesky(H_k) # cholesky decomposition of H_k
           
            alfa_k = self.line_search_wolfe(x_k, s_k)
            delta_k =alfa_k * s_k
            g_k = np.array(self.opt_problem.gradient_value(x_k))
            x_k = x_k + delta_k
            g_updated = np.array(self.opt_problem.gradient_value(x_k))
            gamma_k =  g_updated - g_k
            
            dTg =  np.dot(np.transpose(delta_k), gamma_k)          
            ddT = np.dot(delta_k, np.transpose(delta_k))             
            HggTH = np.dot(np.dot(H_k, gamma_k),np.transpose(gamma_k)) *H_k
            gTHg = np.dot(np.dot(np.transpose(gamma_k), H_k ), gamma_k)
            
            H_k = H_k  + reg + ddT/ dTg  - HggTH / gTHg
            s_k = np.dot(-H_k,np.array(self.opt_problem.gradient_value(x_k)))
            i= i+1
        print("Number of iterations DFP:", i-1)
        return x_k
        