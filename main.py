from utils.analysis import get_results
from utils.dataloader import load_data, get_XY
from utils.dataloader import split_data
from utils.preprocessing import Preprocessor
from sklearn.linear_model import LogisticRegression

data_path = "data/agaricus-lepiota.data"
train_set_fraction = 0.6


def main():
    data = load_data(data_path)
    X, Y = get_XY(data)

    preprocessor = Preprocessor(X)
    preprocessor.preprocess_data()

    # print(preprocessor.data)
    # print(preprocessor.data[:, 0])

    X = preprocessor.data

    X_train, Y_train, X_test, Y_test = split_data(X, Y, train_set_fraction)
    log_reg = LogisticRegression()
    log_reg.fit(X_train.astype(float), Y_train)
    print(get_results(log_reg, X_test, Y_test))


if __name__ == '__main__':
    main()
