import numpy as np


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
        self.data = np.array(data)

    def preprocess_data(self):
        """Convert string labels to numerical values in whole data set"""
        for feature in range(len(self.data[0])):
            converter = LabelConverter()
            converter.create_label_dict(self.data[:, feature])
            self.data[:, feature] = converter.convert_to_float(self.data[:, feature])
        self.data = self.data.astype(float)