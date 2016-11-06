import numpy as np

from utils.dataloader import load_data
from utils.dataloader import split_data
from utils.preprocessing import Preprocessor
from sklearn.linear_model import LogisticRegression

from utils.visualization import plot_results

data_path = "data/agaricus-lepiota.data"
train_set_fraction = 0.8


def train_and_test(X, Y, train_set_fraction):
    """Split dataset to train and test part and return its efficacy"""
    X_train, Y_train, X_test, Y_test = split_data(X.astype(float), Y, train_set_fraction)

    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train.astype(float), Y_train)
    return log_reg.score(X.astype(float),Y)


def prepare_data():
    """Load and preprocess data"""
    data = load_data(data_path)

    prep = Preprocessor(data)

    prep.preprocess_data()
    prep.shuffle_data()
    prep.encode_data()

    return prep.X, prep.Y


def measure_fraction_impact(X, Y):
    """Measure impact of train/test fraction on efficacy of algorithm"""
    fraction = train_set_fraction

    fractions = np.array(list(range(1, 100))) / 100
    results_for_fractions = [train_and_test(X, Y, fraction) for fraction in fractions]
    plot_results(fractions, results_for_fractions)


def main():
    X, Y = prepare_data()
    measure_fraction_impact(X, Y)


if __name__ == '__main__':
    main()
