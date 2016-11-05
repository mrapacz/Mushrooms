import numpy as np


def load_data(file_name):
    """Load and shuffle initial data from file and return it"""

    with open(file_name, "r") as file:
        data = np.array([line.rstrip().split(",") for line in file.readlines()])
    return data


def split_data(X, Y, fraction):
    train = int(len(X) * fraction)
    X_train, Y_train = X[:train, :], Y[:train]
    X_test, Y_test = X[train:, :], Y[train:]

    return X_train, Y_train, X_test, Y_test
