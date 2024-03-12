import matplotlib.pyplot as plt


def plot_function_by_sampled_points(x, y, title, save_file=None):
    # Plot the function
    plt.plot(x, y)  # You can adjust the label accordingly
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    if save_file:
        plt.savefig(save_file)
    # plt.show()


def plot_comparison(x1, y1, x2, y2, title, plot_together=True, save_file=None):
    if plot_together:
        plt.plot(x1, y1, label='Expected')
        plt.plot(x2, y2, label='Results of model')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend()
        if save_file:
            plt.savefig(save_file)
        # plt.show()
    else:
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1)
        plt.title('Antiderivative of function')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(2, 1, 2)
        plt.plot(x2, y2)
        plt.title('Model prediction of antiderivative')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.suptitle(title)
        if save_file:
            plt.savefig(save_file)
        # plt.show()

    # plt.show()