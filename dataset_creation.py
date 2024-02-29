import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pickle
import helper

# Given by tasks some we can play around with

NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 200
NUM_TEST_SAMPLES = 200
UNIFORM_DOMAIN = [-1, 1]
NORMAL_DIST_DOMAIN = [0, 1]
GRID_START = 0
GRID_END = 1
MAX_DEGREE = 4
NUM_POINTS = 40

NUM_FUNCTIONS = 100
NUM_RANDOM_POINTS = 10


def sample_points(function):
    x = np.linspace(GRID_START, GRID_END, NUM_POINTS)
    # sample instead
    # x = np.random.uniform(GRID_START, GRID_END, NUM_POINTS)
    y = function(x)
    return x, y


def generate_coeff(method="normal"):
    coeff = []
    for i in range(MAX_DEGREE + 1):
        if method == "uniform":
            coeff.append(np.random.uniform(UNIFORM_DOMAIN[0], UNIFORM_DOMAIN[1]))
        elif method == "normal":
            coeff.append(np.random.normal(0, 1))
    return coeff


def create_datapoints_same_fun():
    coeff = generate_coeff()
    function = np.poly1d(coeff)
    x, y = sample_points(function)
    y = y.tolist()
    integral_coeff = np.polyint(coeff)
    integral_fun = np.poly1d(integral_coeff)

    datapoints = []
    for i in range(NUM_RANDOM_POINTS):
        random_point = np.random.uniform(GRID_START, GRID_END)
        random_point_integral_value = integral_fun(random_point)
        datapoints.append([y, random_point, random_point_integral_value])

    return datapoints


def create_datapoints():
    datapoints = []
    for i in range(NUM_FUNCTIONS):
        datapoints.extend(create_datapoints_same_fun())
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
