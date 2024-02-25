import matplotlib.pyplot as plt


def plot_function_by_sampled_points(x, y):
    # Plot the function
    plt.plot(x, y)  # You can adjust the label accordingly
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of function')
    plt.grid(True)
    plt.show()