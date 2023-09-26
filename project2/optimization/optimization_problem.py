#the function and gradient should be a method that can take the input of an 
#np.array containing the variables. We might want to add assert/try catch 
#statments incase the input does not match the expectations. 

import numpy as np
#from methods import OptimizationMethod


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
    
    
    
    
    #Since the gradient arguement is optional, we need an aproximation of it 
    #It's a simple derivate approximation, if it's not accurate enough a full
    #Tayler approximation should probably be used. 
    def gradient_aprox(self,x): 
        
        a = self.obj_function(x)
        b = np.array([])
        g_aprox = np.array([])
        x_temp = x
        for i in range(len(x)):
            x_temp[i]= x_temp[i] - self.step_size
            b = np.append(b,self.obj_function(x_temp))
            x_temp[i]= x_temp[i] + self.step_size
        #b = np.reshape(b,(len(x)))
        for i in range(len(x)):
            temp_res = (a-b[i])/(self.step_size)
            g_aprox = np.append(g_aprox,temp_res)
            
        return g_aprox
