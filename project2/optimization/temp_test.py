# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:49:55 2023

@author: linat
"""

import numpy as np

from scipy.optimize import basinhopping

import scipy.optimize as so


from optimization_problem import OptimizationProblem
from methods import OptimizationMethod

#test functions for the hessian approximation. 
def x_raised_two_function(x):
    return np.array([x**2])

def grad_x_raised_two_function(x):
    return np.array([2*x])

def multi_variable(x):
    res = 0
    for i in range(len(x)):
        res = x[i]**2 +res
    return np.array([res])

def grad_multi_variable(x):
    res = np.array([])
    for i in range(len(x)):
        res = np.append(res,[2*x[i]])
    return res


x = (2,3)

res = multi_variable(x)

grad_res = grad_multi_variable(x)

stopping_criteria = 1e-10
max_iteration = 2


problem = OptimizationProblem(x_raised_two_function,grad_x_raised_two_function)

hessian_test = OptimizationMethod(problem)

h_aprox_test = hessian_test.hessian_aprox(np.array([2], dtype='float64'),0.01)

problem_2 = OptimizationProblem(multi_variable,grad_multi_variable)

hessian_test_2 = OptimizationMethod(problem_2)

h_aprox_2 = hessian_test_2.hessian_aprox(np.array([2,2,2], dtype='float64'),0.01)

minimum = hessian_test.classical_Newton(np.array([3],dtype='float64'), 0.01)

minimum_2 = hessian_test_2.classical_Newton(np.array([3,3,3],dtype='float64'), 0.01)

minimum_l = hessian_test.Newton_with_exact_line_search(np.array([3],dtype='float64'), 0.01)

minimum_2_l = hessian_test_2.Newton_with_exact_line_search(np.array([3,3,3],dtype='float64'), 0.01)

x_init = np.array([3], dtype='float64')



iteration = 0 
x= x_init
step_size = 0.01
test = 0

p =problem.gradient_value(x_init)

while (iteration<max_iteration and np.all(problem.gradient_value(x)>stopping_criteria)):
    G = hessian_test.hessian_aprox(x, step_size)
    s = - np.dot(np.linalg.inv(G),problem.gradient_value(x))
    x = np.reshape(np.add(x,s),(len(x)))
    iteration = iteration + 1
    test = 1
    p = problem.gradient_value(x)



x = lambda a : a + 10

x = np.array([2,2])

y = lambda a : multi_variable(a*x)[0]

k = y(2)

maxIters=50
stopping_criteria=1e-20
step_size=0.00001
line_search = 'exact_line_search'
x_init = np.array([3,3,3],dtype='float64')

    
iteration = 0 
x= x_init
g = problem_2.gradient_value(x)

#while (np.all(g>stopping_criteria) or iteration<maxIters):
G = hessian_test_2.hessian_aprox(x, step_size)
H = np.linalg.inv(G)
a = so.minimize_scalar(lambda a : problem_2.function_value(x-a*np.dot(H,g))[0]) if line_search == 'exact_line_search' else 1
s = - a.x*np.dot(H,problem_2.gradient_value(x))
x = np.reshape(np.add(x,s),(len(x)))
g = problem_2.gradient_value(x)
x = 1
#iteration = iteration + 1




