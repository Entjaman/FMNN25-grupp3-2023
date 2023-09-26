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