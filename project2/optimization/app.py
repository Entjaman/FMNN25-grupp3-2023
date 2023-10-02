from __future__ import print_function, unicode_literals
from optimization_problem import OptimizationProblem
import chebyquad_problem as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from methods import OptimizationMethod
import methods
import quasi_methods as quasi

from PyInquirer import style_from_dict, Token, prompt
from pprint import pprint

##### PROMPT SETTINGS #####

style = style_from_dict({
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

questions = [
    {
        'type': 'list',
        'name': 'problem',
        'message': 'What problem do you want to solve?',
        'choices': ['Rosenbrock', 'Chebyquad'],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'list',
        'name': 'optimizeMethod',
        'message': 'What Newton method would you like to use?',
        'choices': ['Classical Newton Method', 'Good Broyden', 'Bad Broyden', 'Symmetric Broyden', 'DFP', 'BFGS'],
        'filter': lambda val: val.lower()
    },
    {
        'type': 'list',
        'name': 'lineSearchMethod',
        'message': 'What line search method do you want to use for Classical Newton Method?',
        'choices': ['None (alpha equals one)', 'Exact line search', 'Inexact line search'],
        'when': lambda answers: answers['optimizeMethod'] == 'classical newton method',
        'filter': lambda val: val.lower()
    },
]

##### END OF PROMPT SETTINGS #####


def main():
    # prompts
    print("Welcome to the optimizer!\n")
    print("• If choosing the Rosenbrock function, the initial point is (0,1). A plot will also be generated with exact line search for task 5.")
    print("• Else if choosing the Chebyquad function, the initial point is np.linspace(0,1,11).")
    print("• All Quasi-Newton methods use inexact line search.\n")
    answers = prompt(questions, style=style)
    problem = answers['problem']
    optimize_method = answers['optimizeMethod']
    print("\n\nCalculating...\n\n")
    # end of prompts


    if problem == 'chebyquad':
        op = OptimizationProblem(cp.chebyquad,cp.gradchebyquad)
        xk=np.linspace(0,1,11) # Hardcoded entries
    else: # rosenbrock
        plot()
        op = OptimizationProblem(methods.rosenbrock_function)
        xk = np.array([0.0, 1.0]) # Hardcoded entries


    print("=====================")
    if optimize_method == 'classical newton method':
        om = OptimizationMethod(op)
        match answers['lineSearchMethod']:
            case 'none (alpha equals one)':
                result = om.classical_newton(xk)
                print(f"Result for Classical Newton Method with alpha equals one: {result}")
            case 'exact line search':
                result = om.newton_with_exact_line_search(xk)
                print(f"Result for Classical Newton Method with exact line search: {result}")
            case 'inexact line search':
                result = om.newton_with_inexact_line_search(xk)
                print(f"Result for Classical Newton Method with inexact line search: {result}")
            case _:
                print("Well this is awkward... This shouldn't happen.")

    quasi_newton = quasi.QuasiNewtonMethod(op)
    if optimize_method == 'good broyden':
        result = quasi_newton.minimize_good_broyden(xk)
        print(f"Result for Good Broyden: {result}")
    elif optimize_method == 'bad broyden':
        result = quasi_newton.minimize_bad_broyden(xk)
        print(f"Result for Bad Broyden: {result}")
    elif optimize_method == 'symmetric broyden':
        result = quasi_newton.minimize_symmetric_broyden(xk)
        print(f"Result for Symmetric Broyden: {result}")
    elif optimize_method == 'dfp':
        result = quasi_newton.minimize_DFP(xk)
        print(f"Result for DFP: {result}")
    elif optimize_method == 'bfgs':
        result, hessians = quasi_newton.minimize_bfgs(xk, save_H=True)
        print(f"Result for BFGS: {result}")
        #print(f"Calculated hessians: {hessians}")

    print("=====================")
    reference_value = opt.fmin_bfgs(cp.chebyquad, xk)
    print(f"Scipy optimized result: {reference_value}")
    print("=====================")

    distance = np.abs(result - reference_value)
    print(f"Distance between our result and scipy's result: {distance}")
    


def plot():
    """
    Helper method to plot the Rosenbrock function with exact line search
    """
    
    x = np.array([0.0, 1.0])
    rosenbrock_function = methods.rosenbrock_function
    op = OptimizationProblem(rosenbrock_function)
    om = OptimizationMethod(op)
    # Store optimization path for plotting
    optimization_path = [x.copy()]

    while True:
        x_new = om.newton_with_exact_line_search(
            x
        )  # change this to the method that should run
        optimization_path.append(x_new)
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new

    optimization_path = np.array(optimization_path)

    print("====== TASK 5 ======\n")
    print("Our Solution:", x)
    print("Our Minimum Value:", rosenbrock_function(x))

    result = opt.minimize(rosenbrock_function, x, method="BFGS")
    solution = result.x
    minimum_value = result.fun

    print("Scipy Solution: (BFGS)", solution)
    print("Scipy Minimum Value: (BFGS)", minimum_value)
    print("\nClose plots to continue...")
    print("\n====================\n")

    x = np.linspace(-0.5, 2, 500)
    y = np.linspace(-1.5, 4, 500)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_function([X, Y])

    # Create the contour plot
    contour_levels = 30
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z**0.33, contour_levels, colors="black")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock function")

    # Plot the contour plot in black and white
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, colors="black", levels=np.logspace(0, 5, 35))
    plt.clabel(contour, inline=1, fontsize=10)

    # Plot the optimization path as a black line
    plt.plot(
        optimization_path[:, 0],
        optimization_path[:, 1],
        color="black",
        marker="o",
        markersize=5,
        linewidth=0.5,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Steps of Newton's method to compute a minimum")
    plt.show()


if __name__ == "__main__":
    main()