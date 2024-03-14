import numpy as np
import torch
import torch.cuda

import helper
from model import IntegralModel
import matplotlib.pyplot as plt
import dataset_creation


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


def sample_points(function, grid, num_points):
    x = np.linspace(grid[0], grid[1], num_points)
    y = function(x)
    return x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def anti_derivative_sigmoid(x):
    return np.log(1 + np.exp(x))


# load model

def calculate_mse(pred_antiderivative, antiderivative, interval):
    true_antiderivative_x, true_antiderivative_y = sample_points(antiderivative, interval, 40)
    # Also substract F(0) for the true_antiderivative
    true_antiderivative_y = [z - true_antiderivative_y[0] for z in true_antiderivative_y]
    true_antiderivative = (true_antiderivative_x, true_antiderivative_y)
    #mse = ((true_antiderivative_y - pred_antiderivative[1]) ** 2).mean()
    mse = np.square(np.subtract(true_antiderivative_y, pred_antiderivative[1])).mean()
    return mse


def compare_to_taylor(fun, interval):
    # TODO: compare prediction to taylor approximation up to degree n

    # Compute Taylor expansion coefficients
    center_point = (interval[1] + interval[0]) / 2
    taylor_coefficients = np.polyfit(np.arange(MAX_DEGREE + 1), fun(center_point + np.arange(MAX_DEGREE + 1)),
                                     MAX_DEGREE)
    taylor_expansion = np.poly1d(taylor_coefficients)

    # Sample points for the Taylor expansion
    taylor_x = np.linspace(interval[0], interval[1], 40) # same as fun[0]
    taylor_y = taylor_expansion(taylor_x)

    taylor_approximation = taylor_x, taylor_y

    return taylor_approximation



def test_other_fun(model, fun, antiderivative, fun_name, interval):
    fun = sample_points(fun, interval, 40)
    true_antiderivative_x, true_antiderivative_y = sample_points(antiderivative, interval, 1000)
    # Also substract F(0) for the true_antiderivative
    true_antiderivative_y = [z - true_antiderivative_y[0] for z in true_antiderivative_y]
    true_antiderivative = (true_antiderivative_x, true_antiderivative_y)
    x = fun[0]
    y = fun[1]
    pred_antiderivative_x = x
    pred_antiderivative_y = []
    for i in x:
        res = model(torch.tensor(np.append(y, i)).to(model.device))
        pred_antiderivative_y.append(res.item())

    pred_antiderivative = (pred_antiderivative_x, pred_antiderivative_y)

    mse = calculate_mse(pred_antiderivative, antiderivative, interval)

    taylor_approximation = compare_to_taylor(fun, interval)

    helper.plot_all(fun, true_antiderivative, pred_antiderivative,taylor_approximation,
                    f"{fun_name}: function & antiderivative & pred_antiderivative & Taylor Expansion ",
                    mse,save_file="gen_img/" + fun_name)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    trained_model = IntegralModel.load_from_checkpoint("trained_model/new_version/epoch=9999-step=630000.ckpt")
    trained_model.to(device)
    trained_model.eval()

    test_other_fun(trained_model, np.cos, np.sin, "cos", [0, 1])

    random_fun_coeff = np.random.normal(size=5 + 1)
    random_fun = np.poly1d(random_fun_coeff)
    random_fun_integral = np.poly1d(np.polyint(random_fun))

    test_other_fun(trained_model, np.exp, np.exp, "exp", [0, 1])
