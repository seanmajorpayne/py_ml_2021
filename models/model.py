import numpy as np
from loss import Loss
from utils import classification_rate

class Model:
    """
    Base class for Machine Learning Models
    """
    def __init__(self):
        self.W = None
        self.b = 0
        self.history = {}
        self.predictions = {}
        pass

    def compile(self, optimizer=None, loss=Loss.RSQUARED):
        self.loss = loss
        pass

    def fit(self, x_train, y_train, validation_data, epochs=1000):
        pass

    def cost(self, P, Y):
        """
        Takes predictions and targets and returns the error based
        on the chosen loss function provided at the compile stage.
        :param P: Predictions - Numpy Matrix
        :param Y: Targets - Numpy Matrix
        :return: Cost - Float
        """

        # Mean Squared Error Standardized
        if self.loss == Loss.RSQUARED:
            ssr = Y - P
            sst = Y - np.mean(Y)
            r2 = 1 - (ssr.dot(ssr) / sst.dot(sst))
            return r2

        # Binary Crossentropy
        elif self.loss == Loss.BCE:
            return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))

        # Categorical Crossentropy
        elif self.loss == Loss.CC:
            total = Y * np.log(P)
            return total.sum()

    def score(self, y_train, y_test):
        p_train = np.round(self.predictions["Training Predictions"])
        p_test = np.round(self.predictions["Test Predictions"])
        train_classification = classification_rate(p_train, y_train)
        test_classification = classification_rate(p_test, y_test)
        print("Final train classification rate: {}".format(train_classification))
        print("Final test classification rate: {}".format(test_classification))


