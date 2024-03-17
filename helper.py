import matplotlib.pyplot as plt


def plot_all(fun, antiderivative, antiderivative_pred, title,
             mse, save_file=None):
    x_fun, y_fun = fun
    x_antiderivative, y_antiderivative = antiderivative
    x_antiderivative_pred, y_antiderivative_pred = antiderivative_pred

    plt.figure(figsize=(10, 6))
    plt.plot(x_fun, y_fun, label='Original Function')
    plt.plot(x_antiderivative, y_antiderivative, label='True Antiderivative')
    plt.plot(x_antiderivative_pred, y_antiderivative_pred, label='Predicted Antiderivative')

    plt.title(title + " mse: " + str(mse))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if save_file:
        plt.savefig(save_file)
    plt.show()