import numpy as np

from utils.analysis import get_results
from utils.dataloader import load_data
from utils.dataloader import split_data
from utils.preprocessing import Preprocessor
from sklearn.linear_model import LogisticRegression

data_path = "data/agaricus-lepiota.data"
train_set_fraction = 0.8


def train_and_test(X, Y, train_set_fraction):
    X_train, Y_train, X_test, Y_test = split_data(X.astype(float), Y, train_set_fraction)

    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, Y_train)
    return log_reg.score(X_test, Y_test)


def prepare_data():
    data = load_data(data_path)

    prep = Preprocessor(data)

    prep.preprocess_data()
    prep.shuffle_data()
    prep.encode_data()

    return prep.X, prep.Y


def main():
    fraction = train_set_fraction

    X, Y = prepare_data()
    # fractions = np.array(list(range(1, 100))) / 1000
    # results_for_fractions = [for fraction in fractions]
    res = train_and_test(X, Y, fraction)
    print(res)


if __name__ == '__main__':
    main()
