import numpy as np


def get_results(logistic_regression, X_test, Y_test):
    """Use logistic regression to predict values for test set and compare with Y_test"""

    prediction = logistic_regression.predict(X_test)
    np.set_printoptions(threshold=np.nan)

    print(list(zip(prediction,Y_test)))

