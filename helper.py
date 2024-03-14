import matplotlib.pyplot as plt


def plot_all(func, antiderivative, antiderivative_pred, taylor_approximation, title,
             mse, taylor: bool = False, save_file=None):
    x_func, y_func = func
    x_antiderivative, y_antiderivative = antiderivative
    x_antiderivative_pred, y_antiderivative_pred = antiderivative_pred
    x_taylor, y_taylor = taylor_approximation

    plt.figure(figsize=(10, 6))
    plt.plot(x_func, y_func, label='Original Function')
    plt.plot(x_antiderivative, y_antiderivative, label='True Antiderivative')
    plt.plot(x_antiderivative_pred, y_antiderivative_pred, label='Predicted Antiderivative')
    plt.plot(x_taylor, y_taylor, label='taylor approximation')

    plt.title(title + " mse: " + str(mse))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if save_file:
        plt.savefig(save_file)
    plt.show()