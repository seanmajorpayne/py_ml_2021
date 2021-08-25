import numpy as np
from enum import Enum

class Loss(Enum):
    MSE = 0
    BCE = 1
    CC = 2
    SCC = 3


class Model:
    """
    Base class for Machine Learning Models
    """
    def __init__(self):
        pass

    def compile(self, optimizer=None, loss='mse'):
        pass

    def fit(self, x_train, y_train, validation_data, epochs=1000):
        pass

    def cost(self, P, Y):
        pass
