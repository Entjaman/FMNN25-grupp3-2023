
def plot(optimization_path):
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


def main():
    x = np.array([0.0, 1.0])
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

    print("Our Solution:", x)
    print("Our Minimum Value:", rosenbrock_function(x))

    result = opt.minimize(rosenbrock_function, x, method="BFGS")
    solution = result.x
    minimum_value = result.fun

    print("Scipy Solution: (BFGS)", solution)
    print("Scipy Minimum Value: (BFGS)", minimum_value)

    plot(optimization_path)


if __name__ == "__main__":
    main()