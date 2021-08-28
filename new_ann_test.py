import numpy as np
import matplotlib.pyplot as plt
from models.ann import ANN
from models.dense import Dense
from utils.loss import Loss
from utils.utils import get_data, normalize_data, create_indicator_matrix, classification_rate


Xtrain, Xtest, Ytrain_temp, Ytest_temp = get_data()
Xtrain, Xtest = normalize_data(Xtrain, Xtest)

K = len(set(Ytrain_temp))

Ytrain = create_indicator_matrix(Ytrain_temp)
Ytest = create_indicator_matrix(Ytest_temp)

_, D = Xtrain.shape

M = 500
D = Xtrain.shape[1]
K = len(set(Ytrain_temp))

model = ANN([
    Dense(D, 500),
    Dense(500, 500),
    Dense(500, 500),
    Dense(500, K, activation='softmax')
])

model.compile(optimizer=None, loss=Loss.CC)
model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=200)

print("Train Classification Rate: {}".format(classification_rate(model.predictions["train"], Ytrain)))
print("Test Classification Rate: {}". format(classification_rate(model.predictions["test"], Ytest)))

plt.plot(model.history["train costs"], label="train costs")
plt.plot(model.history["test costs"], label="test costs")
plt.legend()
plt.show()



