#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

SEED = 3
np.random.seed(SEED)  # fix seed for debugging

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - x ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def rand_matrix(w, h, offset=0.25):
    return (2 * np.random.random((w, h)) - 1) * offset


class NeuralNetwork:

    def __init__(self, insize, hidden, outsize, activation='tanh'):
        act_func = {
            'sigmoid': (sigmoid, sigmoid_deriv),
            'tanh': (tanh, tanh_deriv)
        }
        self.act, self.act_deriv = act_func[activation]

        self.weights = [rand_matrix(insize + 1, hidden + 1),
                        rand_matrix(hidden + 1, outsize)]

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        datasize = X.shape[0]
        X = np.column_stack((X, np.ones(datasize))) # one column for bias
        y = np.array(y) # copy

        for k in xrange(epochs):
            chosen = np.random.randint(datasize)
            a = [X[chosen]]

            for i in xrange(len(self.weights)):
                input = np.dot(a[i], self.weights[i])
                a.append(self.act(input))

            error = y[chosen] - a[-1]
            delta = [error * self.act_deriv(a[-1])]

            # from the second to the last layer
            for l in xrange(len(a) - 2, 0, -1):
                hidden_error = delta[-1].dot(self.weights[l].T)
                activated = self.act_deriv(a[l])
                delta.append(activated * hidden_error)
            delta.reverse()

            for i in xrange(len(self.weights)):
                layer = np.atleast_2d(a[i])
                d = np.atleast_2d(delta[i])
                # wi,j = wi, j + a * ai * delta[j]
                self.weights[i] += learning_rate * layer.T.dot(d)

            if k % 10000 == 0:
                print 'epochs:', k
                print 'error:', error

    def predict(self, x):
        a = np.hstack((x, [1]))
        # x = np.array(x)
        # temp = np.ones(x.shape[0] + 1)
        # temp[0:-1] = x
        # a = temp
        for i in xrange(len(self.weights)):
            a = self.act(np.dot(a, self.weights[i]))
        return a
