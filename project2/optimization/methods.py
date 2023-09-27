import numpy as np
import scipy.optimize as so


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
    
    #Don't know why I pick this step_size should maybe find a reason?
    def classical_Newton(self,x_init,step_size=0.00001,stopping_criteria=1e-20, maxIters=50):
        
        return self.Newton_help_method(x_init, 'classic',step_size,stopping_criteria,maxIters)
    
    
    def Newton_with_exact_line_search(self,x_init,step_size=0.00001,stopping_criteria=1e-20, maxIters=50):
        return self.Newton_help_method(x_init, 'exact_line_search',step_size,stopping_criteria,maxIters)
        
        
        return None
    
    
    def Newton_help_method(self,x_init,line_search,step_size=0.00001,stopping_criteria=1e-20, maxIters=50):
        
        iteration = 0 
        x= x_init
        g = self.opt_problem.gradient_value(x)

        while (np.all(g>stopping_criteria) or iteration<maxIters):
            G = self.hessian_aprox(x, step_size)
            H = np.linalg.inv(G)
            a = so.minimize_scalar(lambda a : self.opt_problem.function_value(x-a*np.dot(H,g))[0]).x if line_search == 'exact_line_search' else 1
            s = - a*np.dot(H,self.opt_problem.gradient_value(x))
            x = np.reshape(np.add(x,s),(len(x)))
            g = self.opt_problem.gradient_value(x)
            #x = 1
            iteration = iteration + 1
            
        return x
    
    
    