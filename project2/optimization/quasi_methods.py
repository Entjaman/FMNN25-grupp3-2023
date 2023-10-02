import numpy as np
from scipy import optimize

from methods import OptimizationMethod
from optimization_problem import OptimizationProblem
import chebyquad_problem as cp
import scipy.optimize as so
from  scipy import dot,linspace


class QuasiNewton(OptimizationMethod):
    def __init__(self, gradient_function, G=np.identity(2), tol=1e-20, maxIters=50):
        self.gradient_function = gradient_function
        self.G = G
        self.tol = tol
        self.maxIters = maxIters

    def good_broyden_minimize(self, x):
        """Combines the benefits of Broyden's Rank-1 update with
        Sherman-Morrison's formula for numerical stability and faster
        convergence."""
        iterations = 0
        gradient = self.gradient_function(x)
        G = self.G

        # Check the frobenius norm and max iterations
        while np.linalg.norm(gradient) > self.tol and iterations < self.maxIters:
            # Solve for dx using the current Jacobian G
            G_reg = G + 1e-6 * np.eye(G.shape[0])
            dx = np.linalg.solve(G_reg, -gradient)
            x = x + dx
            gradient_new = self.gradient_function(x)
            print(gradient_new)
            dr = gradient_new - gradient

            # Update the approximation of Jacobian matrix using Sherman-Morrison's formula
            G = G + np.outer(dr - np.dot(G, dx), dx) / np.dot(dx, dx)

            gradient = gradient_new
            iterations += 1

        print('Good broyden nbr iter: ', iterations)
        return x

    def bad_broyden_minimize(self, x):
        """Called bad broyden since it is not as efficient or robust as the
        full Broyden update"""
        iterations = 0
        #  The Jacobian of the gradient is called Hessian and is symmetric.
        G_reg = self.G + 1e-6 * np.eye(self.G.shape[0])
        H = np.linalg.inv(G_reg)
        gradient = self.gradient_function(x)

        # Check the frobenius norm and max iterations
        while np.linalg.norm(gradient) > self.tol and iterations < self.maxIters:
            # Compute the search direction
            s = -np.dot(H, gradient)

            # Update the current solution
            x_new = x + s

            #  Compute the change in gradient
            new_gradient = self.gradient_function(x_new)
            dr = new_gradient - gradient

            # Update the inverse Hessian approximation using Broyden rank-1 update
            H += np.outer(s - np.dot(H, dr), np.dot(H, dr)) / np.dot(
                np.dot(H, dr), np.dot(H, dr)
            )
            gradient = new_gradient
            iterations += 1
            x = x_new

        print('Bad broyden nbr iter: ', iterations)
        return x
    
    def symmetric_broyden_minimize(self, x):
        """Symmetric Broyden update for Jacobian approximation.
         the symmetric Broyden method may have some stability issues, 
         especially when dealing with ill-conditioned or highly nonlinear 
         problems. It might converge slowly or fail to converge in some cases.
         """
        iterations = 0
        G_reg = self.G + 1e-6 * np.eye(self.G.shape[0])
        H = np.linalg.inv(G_reg)
        gradient = self.gradient_function(x)

        while np.linalg.norm(gradient) > self.tol and iterations < self.maxIters:
            # Calculate the direction vector
            s = -np.dot(H, gradient)
            x_new = x + s
            new_gradient = self.gradient_function(x_new)
            dr = new_gradient - gradient
            dx = x_new -x

            # Symmetric Broyden update - ensures that H remains symmetric during the iterations 
            H += np.outer(dx - np.dot(H, dr), dx) / np.dot(dr, dx)

            gradient = new_gradient
            iterations += 1
            x = x_new

        print('Symmetric broyden nbr iter: ', iterations)
        return x


def main():
   
    x = np.array([0.0, 0.0, 0.0])
    op = OptimizationProblem(cp.chebyquad, cp.gradchebyquad)
    om = OptimizationMethod(op)
    G = om.hessian_aprox(x, 1)

    # G needs to be initialized as a good approximation for the methods to work
    # therefore we need to use the hessian_aprox function here, until then this aprox
    # of the Hessian gives an ok approximation.
    #gradient_function = objective_function
    QN = QuasiNewton(cp.gradchebyquad, G,  maxIters=100)

    x_g = QN.good_broyden_minimize(x)
    x_b = QN.bad_broyden_minimize(x)
    x_s = QN.symmetric_broyden_minimize(x) 

    print("Good Broyden")
    print("x:", [i for i in x_g])

    print("\n\nBad Broyden")
    print("x:", [i for i in x_b])

    print("\n\nSymmetric Broyden")
    print("x:", [i for i in x_s],  "\n")

    x_so = so.fmin_bfgs(cp.chebyquad, x)

    print('sicpy opt ', x_so) 


    print('cheby fcn: ', cp.chebyquad_fcn(x_g), 'dot product: ', cp.chebyquad(x_g))
    print('cheby fcn: ', cp.chebyquad_fcn(x_b), 'dot product: ', cp.chebyquad(x_g))
    print('cheby fcn: ', cp.chebyquad_fcn(x_s), 'dot product: ', cp.chebyquad(x_s))
    print('cheby fcn: ', cp.chebyquad_fcn(x_so), 'dot product: ', cp.chebyquad(x_so))


if __name__ == "__main__":
    main()
