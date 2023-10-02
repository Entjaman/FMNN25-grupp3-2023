import numpy as np


class OptimizationProblem():
    def __init__(self, objective_function, gradient=None):
        self.obj_function = objective_function
        self.gradient = gradient
        self.step_size = 0.0001
        
    def function_value(self, x):
        return self.obj_function(x)
    

    def gradient_value(self, x):
        if self.gradient is None: 
            grad_value = self.gradient_aprox(x)
        else: 
            grad_value = self.gradient(x)
        return grad_value
    
    
    def gradient_aprox(self,x): 
        
        a = self.obj_function(x)
        b = np.array([])
        g_aprox = np.array([])
        x_temp = x
        for i in range(len(x)):
            x_temp[i]= x_temp[i] - self.step_size
            b = np.append(b,self.obj_function(x_temp))
            x_temp[i]= x_temp[i] + self.step_size
        for i in range(len(x)):
            temp_res = (a-b[i])/(self.step_size)
            g_aprox = np.append(g_aprox,temp_res)
            
        return g_aprox
