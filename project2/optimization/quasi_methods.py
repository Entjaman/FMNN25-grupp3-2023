import numpy as np
from scipy import optimize

# from methods import OptimizationMethod
# from optimization_problem import OptimizationProblem
from autograd import hessian


class QuasiNewton:
    def __init__(self, f_eq, G=np.identity(2), tol=1e-20, maxIters=50):
        self.f_eq = f_eq
        self.G = G
        self.tol = tol
        self.maxIters = maxIters

    def good_broyden_minimize(self, x):
        """Combines the benefits of Broyden's Rank-1 update with
        Sherman-Morrison's formula for numerical stability and faster
        convergence."""
        iterations = 0
        fx = self.f_eq(x)
        G = self.G

        # Check the frobenius norm and max iterations
        while np.linalg.norm(fx) > self.tol and iterations < self.maxIters:
            # Solve for dx using the current Jacobian G
            dx = np.linalg.solve(G, -fx)
            x = x + dx
            fx_new = self.f_eq(x)
            dk = fx_new - fx

            # Update the approximation of Jacobian matrix using Sherman-Morrison's formula
            G = G + np.outer(dk - np.dot(G, dx), dx) / np.dot(dx, dx)

            fx = fx_new
            iterations += 1

        print('Good broyden nbr iter: ', iterations)
        return x

    def bad_broyden_minimize(self, x):
        """Called bad broyden since it is not as efficient or robust as the
        full Broyden update"""
        iterations = 0
        #  The Jacobian of the gradient is called Hessian and is symmetric.
        H = np.linalg.inv(self.G)
        gradient = self.f_eq(x)

        # Check the frobenius norm and max iterations
        while np.linalg.norm(gradient) > self.tol and iterations < self.maxIters:
            # Compute the search direction
            s = -np.dot(H, gradient)

            # Update the current solution
            x_new = x + s

            #  Compute the change in gradient
            new_gradient = self.f_eq(x_new)
            z = new_gradient - gradient

            # Update the inverse Hessian approximation using Broyden rank-1 update
            H += np.outer(s - np.dot(H, z), np.dot(H, z)) / np.dot(
                np.dot(H, z), np.dot(H, z)
            )
            gradient = new_gradient
            iterations += 1
            x = x_new

        print('Bad broyden nbr iter: ', iterations)
        return x
    
    def symmetric_broyden_minimize(self, x):
        """Symmetric Broyden update for Jacobian approximation."""
        iterations = 0
        H = np.linalg.inv(self.G)
        gradient = self.f_eq(x)

        while np.linalg.norm(gradient) > self.tol and iterations < self.maxIters:
            # Calculate the direction vector
            s = -np.dot(H, gradient)
            x_new = x + s
            new_gradient = self.f_eq(x_new)
            dr = new_gradient - gradient
            dx = x_new -x

            # Symmetric Broyden update - ensures that H remains symmetric during the iterations 
            H += np.outer(dx - np.dot(H, dr), dx) / np.dot(dr, dx)

            gradient = new_gradient
            iterations += 1
            x = x_new

        print('Symmetric broyden nbr iter: ', iterations)
        return x


def objective_function(x):
    return np.array([x[0] + 2 * x[1] - 2, x[0] ** 2 + 4 * x[1] ** 2 - 4])


def main():
    x = np.array([1.0, 2.0])

    # op = OptimizationProblem(fs)
    # om = OptimizationMethod(op)
    # G = om.hessian_aprox(x, 1) # - cannot reshape x of len 2....
    # print(G)

    # G =  hessian(fs)(x)
    # print(np.array(G))

    # G needs to be initialized as a good approximation for the methods to work
    # therefore we need to use the hessian_aprox function here, until then this aprox
    # of the Hessian gives an ok approximation.
    G = np.array([[1, 2], [2, 16]])

    QN = QuasiNewton(objective_function, G)

    x_g = QN.good_broyden_minimize(x)
    x_b = QN.bad_broyden_minimize(x)
    x_s = QN.symmetric_broyden_minimize(x) 

    print("Good Broyden")
    print(
        "x and y:", x_g[0], x_g[1], "\ndiff: ", np.abs(0 - x_g[0]), np.abs(1 - x_g[1])
    )

    print("\n\nBad Broyden")
    print(
        "x and y:", x_b[0], x_b[1], "\ndiff: ", np.abs(0 - x_b[0]), np.abs(1 - x_b[1])
    )

    print("\n\nSymmetric Broyden")
    print(
        "x and y:", x_s[0], x_s[1], "\ndiff: ", np.abs(0 - x_s[0]), np.abs(1 - x_s[1])
    )

    ## test example
    sol = optimize.broyden1(objective_function, x)
    print("\n\nGood Broyden Scipy solution", sol)


if __name__ == "__main__":
    main()
