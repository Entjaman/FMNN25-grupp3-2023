import numpy as np
import scipy.optimize as so
from optimization_problem import OptimizationProblem
import scipy.optimize as opt
import matplotlib.pyplot as plt


class OptimizationMethod:
    def __init__(self, opt_problem):
        self.opt_problem = opt_problem

    def hessian_aprox(self, x, step_size):
        a = self.opt_problem.gradient_value(x)
        b = np.array([])
        h_aprox = np.array([])
        x_temp = x
        for i in range(len(x)):
            x_temp = x
            x_temp[i] = x_temp[i] - step_size
            b = np.append(b, self.opt_problem.gradient_value(x_temp))
            x_temp[i] = x_temp[i] + step_size
        b = np.reshape(b, (len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                temp_res = (a[i] - b[i][j]) / (x[i] - (x[i] - step_size))
                h_aprox = np.append(h_aprox, temp_res)
        h_aprox = np.reshape(h_aprox, (len(x), len(x)))

        return h_aprox

    def classical_newton(
        self, x_init, step_size=0.00001, stopping_criteria=1e-20, maxIters=50
    ):
        return self.newton_help_method(
            x_init, "classic", step_size, stopping_criteria, maxIters
        )

    def newton_with_exact_line_search(
        self, x_init, step_size=0.00001, stopping_criteria=1e-20, maxIters=50
    ):
        return self.newton_help_method(
            x_init, "exact_line_search", step_size, stopping_criteria, maxIters
        )

    def newton_with_inexact_line_search(
        self, x_init, step_size=0.00001, stopping_criteria=1e-20, maxIters=50
    ):
        return self.newton_help_method(
            x_init, "inexact_line_search", step_size, stopping_criteria, maxIters
        )

    def newton_help_method(
        self,
        x_init,
        line_search,
        step_size=0.00001,
        stopping_criteria=1e-20,
        maxIters=50,
    ):
        iteration = 0
        x = x_init
        g = self.opt_problem.gradient_value(x)

        while np.all(g > stopping_criteria) or iteration < maxIters:
            G = self.hessian_aprox(x, step_size)
            H = np.linalg.inv(G)
            s = -np.dot(H, g)
            if line_search == "exact_line_search":
                a = so.minimize_scalar(
                    lambda a: self.opt_problem.function_value(x + a * s)
                ).x
            elif line_search == "inexact_line_search":
                a = self.line_search_wolfe(x, s)
            else:
                a = 1
            s_a = a * s
            x = np.reshape(np.add(x, s_a), (len(x)))
            g = self.opt_problem.gradient_value(x)
            iteration = iteration + 1

        return x

    def line_search_wolfe(self, xk, sk, c1=1e-20, c2=0.9, maxIter=50):
        """
        xk                  : the x values at iteration k
        sk                  : search direction = -H * gk at iteration k
        c1                  : arbitrary constant between (0,1)
        c2                  : arbitrary constant for the curvature condition between (c1,1)

        Implemented as per slide 21 in the course lecture
        """
        alfa_k_minus = 1

        fx = np.array(self.opt_problem.function_value(xk))
        dfx = np.array(self.opt_problem.gradient_value(xk))

        def phi(alfa):
            return np.array(self.opt_problem.obj_function(xk + alfa * sk))

        def dphi(alfa):
            return np.array(self.opt_problem.gradient_value(xk + alfa * sk))

        iteration = 0
        # Armijo rule not fullfiled
        while (
            phi(alfa_k_minus) > fx + c1 * alfa_k_minus * np.dot(dfx.T, sk)
            and iteration < maxIter
        ):
            alfa_k_minus = alfa_k_minus / 2
            iteration = iteration + 1
        alfa_k_plus = alfa_k_minus
        iteration = 0
        while (
            phi(alfa_k_plus) <= fx + c1 * alfa_k_plus * np.dot(dfx.T, sk)
            and iteration < maxIter
        ):
            alfa_k_plus = 2 * alfa_k_plus
            iteration = iteration + 1
        iteration = 0
        while (
            np.dot(dphi(alfa_k_minus).T, sk) < c2 * np.dot(dfx.T, sk)
            and iteration < maxIter
        ):
            alfa_zero = (alfa_k_plus + alfa_k_minus) / 2
            if phi(alfa_zero) <= fx + c1 * alfa_zero * np.dot(dfx.T, sk):
                alfa_k_minus = alfa_zero
            else:
                alfa_k_plus = alfa_zero
            iteration = iteration + 1

        return alfa_k_minus


def rosenbrock_function(x):
    return 100 * (x[1] - (x[0]) ** 2) ** 2 + (1 - x[0]) ** 2
