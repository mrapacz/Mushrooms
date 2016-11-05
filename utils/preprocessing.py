import numpy as np
from sklearn.preprocessing import OneHotEncoder


class LabelConverter:
    def __init__(self):
        self.labels = dict()

    def create_label_dict(self, values):
        """Assign integer values to labels"""
        labels = list(set(values))
        self.labels = {labels[i]: i for i in range(len(labels))}

    def convert_to_float(self, values):
        """Convert string labels to their values based on self.dict"""
        return [self.labels[value] for value in values]


class Preprocessor:
    def __init__(self, data):
        """Split to X,Y assuming Y labels are the first feature"""
        self.data = np.array(data)
        self.encoded_data = list()
        self.X = self.data[:, 1:]
        self.Y = self.data[:, 0]

    def preprocess_data(self):
        """Convert string labels to numerical values in whole data set"""

        # starting from 1 so as the first value is the result class, doesn't have to be float
        for feature in range(len(self.X[0])):
            converter = LabelConverter()
            converter.create_label_dict(self.X[:, feature])
            self.X[:, feature] = converter.convert_to_float(self.X[:, feature])

    def shuffle_data(self):
        np.random.shuffle(self.data)

    def encode_data(self):
        n_values = [6, 4, 10, 2, 9, 2, 2, 2, 12, 2, 5, 4, 4, 9, 9, 1, 4, 3, 5, 9, 6, 7]
        enc = OneHotEncoder(n_values=n_values)
        result = enc.fit_transform(self.X)
        print(result.toarray())

        self.encoded_data = result.toarray()
