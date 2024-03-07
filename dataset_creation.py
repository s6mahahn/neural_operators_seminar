import csv
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

np.random.seed(42)

NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 200
NUM_TEST_SAMPLES = 200
UNIFORM_DOMAIN = [-1, 1]
NORMAL_DIST_DOMAIN = [0, 1]
GRID_START = 0
GRID_END = 1
MAX_DEGREE = 4
NUM_POINTS = 40

NUM_RANDOM_POINTS = 1


def sample_points(function):
    x = np.linspace(GRID_START, GRID_END, NUM_POINTS)
    # sample instead
    # x = np.random.uniform(GRID_START, GRID_END, NUM_POINTS)
    y = function(x)
    return x, y


def create_datapoints_same_fun(chebyshev_polymial):
    coeff = np.random.normal(size=MAX_DEGREE+1)
    if chebyshev_polymial:
        x, y, integral_fun = generate_chebyshev_random_polynomial_values()
    else:
        function = np.poly1d(coeff)
        x, y = sample_points(function)
        y = y.tolist()
        integral_coeff = np.polyint(coeff)
        integral_fun = np.poly1d(integral_coeff)

    datapoints = []
    for i in range(NUM_RANDOM_POINTS):
        random_point = np.random.uniform(GRID_START, GRID_END)
        # Here should be F(a)-F(0)
        integral_value_at_random_point = integral_fun(random_point)
        integral_value_at_zero = integral_fun(0)
        actual_antiderivative_at_random_point = integral_value_at_random_point - integral_value_at_zero
        datapoints.append([y, random_point, actual_antiderivative_at_random_point])
    return datapoints


def generate_chebyshev_random_polynomial_values(num_samples=1000):
    coeffs = np.random.normal(size=MAX_DEGREE+1)
    polynom = np.polynomial.chebyshev.Chebyshev(coeffs)
    x_values = np.linspace(GRID_START, GRID_END, NUM_POINTS)
    y_values = polynom(x_values)

    integral = polynom.integ()
    return x_values, y_values, integral


def create_datapoints(total_datapoints, chebyshev_polymial):
    assert total_datapoints % NUM_RANDOM_POINTS == 0, f"number of requested datapoints not a multiple of {NUM_POINTS} "
    num_functions = int(np.ceil(total_datapoints / NUM_RANDOM_POINTS))
    datapoints = []
    for i in range(num_functions):
        datapoints.extend(create_datapoints_same_fun(chebyshev_polymial))
    return datapoints


def write_to_pickle(name, path, data, split):
    extension = ".pickle"
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%y%m%d_%H%M")
    file_path = Path(path) / Path(name + "_" + date_string + "_" + split + extension)

    # Write list to pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def write_to_csv(name, path, data, split):
    extension = ".csv"
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%y%m%d_%H%M")
    file_path = Path(path) / Path(name + "_" + date_string + "_" + split + extension)
    # Write data to CSV file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    # Example usage
    data = create_datapoints(1, True)

    data2 = create_datapoints(1, False)
