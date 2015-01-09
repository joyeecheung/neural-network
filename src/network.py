#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

SEED = 3
OUTPUT = -1
INPUT = 0
INITIAL_MARGIN = 0.25

np.random.seed(SEED)  # fix seed for debugging

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - x ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def rand_matrix(w, h, margin=INITIAL_MARGIN):
    return (2 * np.random.random((w, h)) - 1) * margin


class NeuralNetwork:
    def __init__(self, layers, activator='sigmoid'):
        # 3 layers, input, hidden, output
        act_func = {
            'sigmoid': (sigmoid, sigmoid_deriv),
            'tanh': (tanh, tanh_deriv)
        }
        self.act, self.act_deriv = act_func[activator]

        # 1 ~ last second
        self.weights = [rand_matrix(layers[i - 1] + 1, layers[i] + 1)
                        for i in xrange(1, len(layers) - 1)]
        # last layer, no bias
        self.weights.append(rand_matrix(layers[OUTPUT - 1] + 1, layers[OUTPUT]))

    def learn(self, X, y, epochs=10000, rate=0.2):
        datasize = X.shape[0]
        noninput_layer_count = len(self.weights)

        X = np.column_stack((X, np.ones(datasize))) # one column for bias
        y = np.array(y) # copy

        for k in xrange(epochs):  # iterate until enough epochs
            # propagate randomly instead of iteration
            chosen = np.random.randint(datasize)
            a = [X[chosen]]  # ai = Xi, though here i is random

            # save g(inj) to the end of a
            for j in xrange(noninput_layer_count):
                total = np.dot(a[j], self.weights[j])
                a.append(self.act(total))

            # propagate back
            error = y[chosen] - a[OUTPUT]
            delta = [error * self.act_deriv(a[OUTPUT])]  # output layer

            for l in xrange(len(a) - 2, 0, -1):  # backward
                hidden_error = np.dot(delta[OUTPUT], self.weights[l].T)
                activated = self.act_deriv(a[l])
                delta.append(activated * hidden_error)
            delta.reverse()  # now [input delta, hidden delta, ..., output delta]

            # update weights
            for j in xrange(noninput_layer_count):
                layer = np.atleast_2d(a[j]).T
                d = np.atleast_2d(delta[j])
                self.weights[j] += rate * np.dot(layer, d)

            if k % (epochs / 10) == 0:  # record
                print 'epochs:', k
                print 'error:', error

    def classify(self, x):
        noninput_layer_count = len(self.weights)
        a = np.hstack((x, [1]))
        for i in xrange(noninput_layer_count):
            a = self.act(np.dot(a, self.weights[i]))
        return a
