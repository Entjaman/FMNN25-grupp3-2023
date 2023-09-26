import numpy as np


class OptimizationMethod():
    def __init__(self, opt_problem,stopping_criteria, maxIters):
        self.opt_problem = opt_problem
        self.stopping_criteria = stopping_criteria
        self.maxIters = maxIters
    
    
    #This is the hessian_aprox for the classical newton method
    #Should maybe reside in that class, but am unsure if any
    #Of the Quasi methods use the same aproximation. 
    #Now it's a simple derivate aproximation, might want to be
    #be changed to a Taylor expansion. 
    def hessian_aprox(self,x,step_size):
        
        a = self.opt_problem.gradient_value(x)
        b = np.array([])
        h_aprox = np.array([])
        x_temp = x
        for i in range(len(x)):
            x_temp = x
            x_temp[i]= x_temp[i] - step_size
            b = np.append(b,self.opt_problem.gradient_value(x_temp))
            x_temp[i]= x_temp[i] + step_size
        b = np.reshape(b,(len(x),len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                temp_res = (a[i]-b[i][j])/(x[i]-(x[i]-step_size))
                h_aprox = np.append(h_aprox,temp_res)
        h_aprox = np.reshape(h_aprox,(len(x),len(x)))
        
        return h_aprox
    
    def classical_Newton(self,x_init,step_size):
        
        iteration = 0 
        x= x_init


        while (np.all(self.opt_problem.gradient_value(x)>self.stopping_criteria) or iteration<self.maxIters):
            G = self.hessian_aprox(x, step_size)
            s = - np.dot(np.linalg.inv(G),self.opt_problem.gradient_value(x))
            x = np.reshape(np.add(x,s),(len(x)))
            #x = 1
            iteration = iteration + 1
            
        return x
   
    def line_search_wolfe(self,xk, sk,c1=1e-4,c2=0.9, maxIter=50):
        
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
        #print(iteration)
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