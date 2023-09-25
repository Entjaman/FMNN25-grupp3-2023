# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:49:55 2023

@author: linat
"""

import numpy as np

from scipy.optimize import basinhopping

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

hessian_test = OptimizationMethod(problem,stopping_criteria,max_iteration)

h_aprox_test = hessian_test.hessian_aprox(np.array([2], dtype='float64'),0.01)

problem_2 = OptimizationProblem(multi_variable,grad_multi_variable)

hessian_test_2 = OptimizationMethod(problem_2,stopping_criteria,max_iteration)

h_aprox_2 = hessian_test_2.hessian_aprox(np.array([2,2,2], dtype='float64'),0.01)

minimum = hessian_test.classical_Newton(np.array([3],dtype='float64'), 0.01)

minimum_2 = hessian_test_2.classical_Newton(np.array([3,3,3],dtype='float64'), 0.01)

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





