import numpy as np
import scipy.linalg as sla
from scipy import optimize
from numpy.linalg import inv
#from methods import OptimizationMethod
#from optimization_problem import OptimizationProblem
from autograd import hessian

class GoodBroyden:
    """ Combines the benefits of Broyden's Rank-1 update with 
    Sherman-Morrison's formula for numerical stability and faster 
    convergence."""

    def __init__(self, f_eq, J, tol=1e-10, maxIters=50):
        self.f_eq = f_eq
        self.G = J
        self.tol = tol
        self.maxIters = maxIters

    def good_broyden_minimize(self, x):
        iterations = 0
        fx = self.f_eq(x)
        G = self.G

         # Check the frobenius norm and max iterations
        while np.linalg.norm(fx) > self.tol and iterations < self.maxIters:
            # Solve for dx using the current Jacobian G
            dx = np.linalg.solve(G, - fx)
            x = x + dx
            fx_new = self.f_eq(x)
            dk = fx_new - fx

            # Update the approximation of Jacobian matrix using Sherman-Morrison's formula
            G = G + np.outer(dk - np.dot(G, dx), dx) / np.dot(dx, dx)

            fx = fx_new
            iterations += 1

        return x

    
class BadBroyden:
    """ Called bad broyden since it is not as efficient or robust as the 
    full Broyden update"""

    def __init__(self, gradient_func, G = np.identity(2), tol=1e-10, maxIters=50):
        self.gradient_func = gradient_func
        self.G = G
        self.tol = tol
        self.maxIters = maxIters

    def bad_broyden_minimize(self, x):
        iterations = 0
        #  The Jacobian of the gradient is called Hessian and is symmetric.
        H = inv(self.G)
        gradient = self.gradient_func(x)

        # Check the frobenius norm and max iterations
        while np.linalg.norm(gradient) > self.tol and iterations < self.maxIters:

            # Compute the search direction
            s = -np.dot(H, gradient)

            # Update the current solution
            x_new = x + s

            #  Compute the change in gradient
            new_gradient = self.gradient_func(x_new)
            z = new_gradient - gradient 
    
            # Update the inverse Hessian approximation using Broyden rank-1 update
            H = H + np.outer(s - np.dot(H, z), np.dot(H, z)) / np.dot(np.dot(H, z), np.dot(H, z))
            gradient = new_gradient
            iterations += 1
            x = x_new

        return x



def fs(x):
    return np.array([x[0] + 2 * x[1] - 2, x[0] ** 2 + 4 * x[1] ** 2 - 4])



def main():
    x = np.array([1.0, 2.0])

    # op = OptimizationProblem(fs)
    # om = OptimizationMethod(op)
    # G = om.hessian_aprox(x, 1) # - cannot reshape x of len 2....
    # print(G)

    # G =  hessian(fs)(x)
    # print(np.array(G))

    tol = 1e-15
    
    G = np.array([[1, 2], [2, 16]])

    gb = GoodBroyden(fs, G, tol)
    bb = BadBroyden(fs, G, tol)

    x_g = gb.good_broyden_minimize(x)
    x_b = bb.bad_broyden_minimize(x)
  
    print("Good Broyden")
    print("x and y:", x_g[0], x_g[1], "\ndiff: ", np.abs(0-x_g[0]), np.abs(1-x_g[1]))

    print("\n\nBad Broyden")
    print("x and y:", x_b[0], x_b[1], "\ndiff: ", np.abs(0-x_b[0]), np.abs(1-x_b[1]))

    ## test example
    sol = optimize.broyden1(fs, x)
    print("\n\nGood Broyden Scipy solution", sol)

if __name__ == "__main__":
    main()
