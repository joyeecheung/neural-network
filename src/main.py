#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from util import get_filenames
from network import NeuralNetwork

SEP = ','

def splitXY(data):
    X = data[:, :-1]
    y = np.array([[1.0 if i == x[0] else 0.0
                   for i in xrange(10)]
                  for x in data[:, -1:]])
    return normalize(X), y


def normalize(data):
    return (data - data.min()) / data.max()


def main():
    files = get_filenames()
    data = np.loadtxt(files.train, delimiter=SEP)
    test = np.loadtxt(files.test, delimiter=SEP)

    X, y = splitXY(data)
    tX, _ = splitXY(test)

    # input size 64, hidden units 50, output size 10
    nn = NeuralNetwork([64, 50, 10], activator='sigmoid')
    nn.learn(X, y, epochs=10000)
    yhat = np.array([np.argmax(nn.classify(e)) for e in tX])
    ytrue = test[:, -1:].ravel()

    print confusion_matrix(ytrue, yhat)
    print classification_report(ytrue, yhat)


if __name__ == "__main__":
    main()
