import numpy as np
import optimization_problem


class OptimizationMethod():
    def __init__(self, opt_problem):
        self.opt_problem = opt_problem
    
    
    #This is the hessian_aprox for the classical newton method
    #Should maybe reside in that class, but am unsure if any
    #Of the Quasi methods use the same aproximation. 
    #Now it's a simple derivate aproximation, might want to be
    #be changed to a Taylor expansion. 
    def hessian_aprox(self,x,step_size):
        
        a = self.opt_problem.gradient_value(x)
        b = np.array([])
        h_aprox = np.array([])
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
    
    
    def minimizer(self):
        
        return None