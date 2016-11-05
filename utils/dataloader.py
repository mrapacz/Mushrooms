import numpy as np


def load_data(file_name):
    """Load initial data from file and return it"""

    with open(file_name, "r") as file:
        data = [line.rstrip().split(",") for line in file.readlines()]
        return np.array(data)


def get_XY(data):
    """Split data to X and Y assuming classification result is first"""

    X = data[:, 1:]
    Y = data[:, 0]

    return X, Y


def split_data(X, Y, fraction):
    train = int(len(X) * fraction)
    X_train, Y_train = X[:train, :], Y[:train]
    X_test, Y_test = X[train:, :], Y[train:]

    return X_train, Y_train, X_test, Y_test
