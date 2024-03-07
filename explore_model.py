import numpy as np
import torch
import torch.cuda
import dataset_creation
import helper
from model import IntegralModel

"""
file for exploring and testing the trained model
    testing things such as:
    - x values outside the trained range (0 to 1)
    - non-polynomial functions such as
        - sin
        - exp
        - logistic function, tanh
        - non-smooth functions
    maybe plotting stuff for these experiments
"""


def sample_points(function, grid):
    x = np.linspace(grid[0], grid[1], 40)
    y = function(x)
    return x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def anti_derivative_sigmoid(x):
    return np.log(1 + np.exp(x))


# load model

def test_other_fun(model, fun, antiderivative, fun_name, interval):
    fun = sample_points(fun, interval)
    true_antiderivative = sample_points(antiderivative, interval)
    x = fun[0]
    y = fun[1]
    pred_antiderivative_x = x
    pred_antiderivative_y = []
    for i in x:
        res = model(torch.tensor(np.append(y, i)).to(model.device))
        pred_antiderivative_y.append(res.item())
    pred_antiderivative = (pred_antiderivative_x, pred_antiderivative_y)
    helper.plot_all(fun, true_antiderivative, pred_antiderivative, title=f"{fun_name}: function & antiderivative", save_file="gen_img/" + fun_name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = IntegralModel.load_from_checkpoint("trained_model/epoch9999-step160000.ckpt")
    trained_model.to(device)
    trained_model.eval()

    test_other_fun(trained_model, np.cos, np.sin, "cos", [0, 1])

    random_fun_coeff = dataset_creation.generate_coeff()
    random_fun = np.poly1d(random_fun_coeff)
    random_fun_integral = np.poly1d(np.polyint(random_fun))

    test_other_fun(trained_model, random_fun, random_fun_integral, "cos", [0, 1])