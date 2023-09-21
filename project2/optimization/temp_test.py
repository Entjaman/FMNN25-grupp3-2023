# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:49:55 2023

@author: linat
"""

import numpy as np

from scipy.optimize import basinhopping

from OptimizationProblem import OptimizationProblem
from Methods import OptimizationMethod

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


problem = OptimizationProblem(x_raised_two_function,grad_x_raised_two_function)

hessian_test = OptimizationMethod(problem)

h_aprox = hessian_test.hessian_aprox(np.array([2], dtype='float64'),0.01)

problem_2 = OptimizationProblem(multi_variable,grad_multi_variable)

hessian_test_2 = OptimizationMethod(problem)

#h_aprox_2 = hessian_test.hessian_aprox(np.array([2,2], dtype='float64'),0.01)

x = np.array([2,2], dtype='float64')
step_size = 0.01

a = problem_2.gradient_value(x)
b = np.array([])
h_aprox = np.array([])
for i in range(len(x)):
    x_temp = x
    x_temp[i]= x_temp[i] - step_size
    b = np.append(b,problem_2.gradient_value(x_temp)[i])
    x_temp[i] = x_temp[i] + step_size
for i in range(len(x)):
    for j in range(len(x)):
        temp_res = (a[i]-b[j])/(x[i]-(x[i]-step_size))
    h_aprox = np.append(h_aprox,temp_res)
#h_aprox = np.reshape(h_aprox,(len(x),len(x)))