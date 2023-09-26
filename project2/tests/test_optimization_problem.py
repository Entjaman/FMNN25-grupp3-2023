import pytest
import optimization_problem as op
import methods

# test functions for the hessian approximation. 
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

