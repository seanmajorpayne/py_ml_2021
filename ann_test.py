import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_data, normalize_data, create_indicator_matrix, relu, softmax, classification_rate

def forward_hidden(X, W1, b1):
    return relu(X.dot(W1) + b1)


def forward_output(Z, W2, b2):
    return softmax(Z.dot(W2) + b2)


def cost(T, Y):
    tot = T * np.log(Y)
    return -tot.mean()


Xtrain, Xtest, Ytrain_temp, Ytest_temp = get_data()
Xtrain, Xtest = normalize_data(Xtrain, Xtest)

K = len(set(Ytrain_temp))

Ytrain = create_indicator_matrix(Ytrain_temp)
Ytest = create_indicator_matrix(Ytest_temp)

_, D = Xtrain.shape

M = 500

layers = 3
L = layers - 1

W = [np.random.randn(M, M) / np.sqrt(M) for i in range(L - 2)]
W.insert(0, np.random.randn(D, M) / np.sqrt(D))
W.append(np.random.randn(M, K) / np.sqrt(M))

b = [np.zeros(M) for i in range(L - 1)]
b.append(np.zeros(K))

Z = [np.random.randn(1, 1) for i in range(L)]

learning_rate = 10e-8
epochs = 500
costs = []
reg = 0.001
for epoch in range(epochs):
    Z[0] = forward_hidden(Xtrain, W[0], b[0])
    for i in range(1, L - 1):
        Z[i] = forward_hidden(Z[i - 1], W[i], b[i])
    Z[L - 1] = forward_output(Z[L - 2], W[L - 1], b[L - 1])
    pYtrain = Z[L - 1]

    delta = pYtrain - Ytrain
    W[L - 1] -= learning_rate * Z[L - 2].T.dot(delta) + reg * W[L - 1]
    b[L - 1] -= learning_rate * delta.sum(axis=0) + reg * b[L - 1]
    for j in reversed(range(1, L - 1)):
        delta = delta.dot(W[j+1].T) * (Z[j] > 0)
        gradient_wH = Z[j - 1].T.dot(delta)
        gradient_bH = delta.sum(axis=0)
        W[j] -= learning_rate * gradient_wH + reg * W[j]
        b[j] -= learning_rate * gradient_bH + reg * b[j]
    delta = delta.dot(W[1].T) * (Z[0] > 0)
    W[0] -= learning_rate * Xtrain.T.dot(delta) + reg * W[0]
    b[0] -= learning_rate * delta.sum(axis=0) + reg * b[0]

    c = cost(Ytrain, pYtrain)
    costs.append(c)

P = np.argmax(pYtrain, axis=1)
print("Final train classification rate: {}".format(classification_rate(np.round(P), Ytrain_temp)))
plt.plot(costs)
plt.show()
