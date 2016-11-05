def get_results(logistic_regression, X_test, Y_test):
    """Use logistic regression to predict values for test set and compare with Y_test"""
    print(X_test.shape)
    print(logistic_regression.predict(X_test.reshape()))
    return ""
