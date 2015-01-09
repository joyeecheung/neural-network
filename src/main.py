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
    data -= data.min()
    data /= data.max()
    return data


def main():
    files = get_filenames()
    data = np.loadtxt(files.train, delimiter=SEP)
    test = np.loadtxt(files.test, delimiter=SEP)

    X, y = splitXY(data)
    tX, ty = splitXY(test)

    nn = NeuralNetwork(insize=64, hidden=50, outsize=10, 'sigmoid')
    nn.fit(X, y, epochs=50000)
    result = np.array([np.argmax(nn.predict(e)) for e in tX])
    yy = test[:, -1:].ravel()

    print confusion_matrix(yy, result)
    print classification_report(yy, result)


if __name__ == "__main__":
    main()
