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

def test_other_fun(trained_model, fun, antiderivative, plot_together, fun_name, interval):
    fun = sample_points(fun, interval)
    true_antiderivative = sample_points(antiderivative, interval)
    x = fun[0]
    y = fun[1]
    pred_antiderivative = []
    for i in x:
        res = trained_model(torch.tensor(np.append(y, i)).to(trained_model.device))
        pred_antiderivative.append(res.item())
    #helper.plot_function_by_sampled_points(fun[0], fun[1], title=fun_name, save_file="gen_img/"+fun_name)
    helper.plot_comparison(fun[0], fun[1], true_antiderivative[0],
                           true_antiderivative[1], title=f"{fun_name}: function & antiderivative",
                           plot_together=plot_together, save_file="gen_img/" + fun_name)
    helper.plot_comparison(true_antiderivative[0], true_antiderivative[1], x,
                           pred_antiderivative, title=f"antiderivative of {fun_name}: true vs pred",
                           plot_together=plot_together, save_file="gen_img/"+fun_name+"_int")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model = IntegralModel.load_from_checkpoint("tb_logs/my_model/version_12/checkpoints/epoch=9999-step=160000.ckpt")
    trained_model.to(device)
    trained_model.eval()

    test_other_fun(trained_model, np.cos, np.sin, False, "cos", [0, 1])
    test_other_fun(trained_model, sigmoid, anti_derivative_sigmoid, False, "sigmoid", [0, 1])
    test_other_fun(trained_model, np.exp, np.exp, False, "exp", [0, 1])


    random_fun_coeff = dataset_creation.generate_coeff()
    random_fun = np.poly1d(random_fun_coeff)
    random_fun_integral = np.poly1d(np.polyint(random_fun))

    test_other_fun(trained_model, random_fun, random_fun_integral, False, "random", [0, 1])
    test_other_fun(trained_model, random_fun, random_fun_integral, False, "random", [-2, -1])


