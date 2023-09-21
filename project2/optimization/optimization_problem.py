#the function and gradient should be a method that can take the input of an 
#np.array containing the variables. We might want to add assert/try catch 
#statments incase the input does not match the expectations. 

import numpy as np
from methods import OptimizationMethod


class OptimizationProblem():
    def __init__(self, objective_function, gradient=None):
        self.obj_function = objective_function
        self.gradient = gradient
        
        
    def function_value(self, x):
        return self.obj_function(x)
    

    def gradient_value(self, x):
        if self.gradient is None: 
            grad_value = self.gradient_aprox(x)
        else: 
            grad_value = self.gradient(x)
        return grad_value
    
    #Since the gradient arguement is optional, we need an aproximation of it 
    #when it's not sent in. It should be returned as a numpy array. 
    def gradient_aprox(self,x): 
        
        return None
    
    #calls the method class, not done. 
    def solve_root(self):
        solver = OptimizationMethod(self)
        
        return None