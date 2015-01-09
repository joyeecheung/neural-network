#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from util import get_filenames
from network import NeuralNetwork

SEP = ','
SAMPLE = 30

def splitXY(data):
    X = data[:, :-1]
    y = np.array([[1.0 if i == x[0] else 0.0
                   for i in xrange(10)]
                  for x in data[:, -1:]])
    return normalize(X), y


def normalize(data):
    return (data - data.min()) / data.max()


def plot_curve(x, y, xlabel, ylabel, name, dest):
    print name
    # statistics
    ymean, ystd, ymin, ymax = np.mean(y), np.std(y), np.min(y), np.max(y)
    print 'Mean of precision = %.4f' % (ymean)
    print 'Standard deviation of precision = %.4f' % (ystd)
    print 'Min = %.4f, max = %.4f' % (ymin, ymax)

    xy = sorted(zip(x, y), key=lambda a: a[0])
    x, y = zip(*xy)

    plt.figure()
    # setup decorations
    plt.rc('font', family='serif')
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plot smoothed learning curve
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = interp1d(x, y)(xnew)
    plt.plot(x, y, '.', xnew, ynew, '--')

    # annotation
    box = dict(boxstyle='square', fc="w", ec="k")
    txt = '$\mu = %.4f$, $\sigma = %.4f$' % (ymean, ystd)
    txt += ', $min = %.4f$, $max = %.4f$' % (ymin, ymax)
    plt.text(170, 0.05, txt, bbox=box)

    plt.savefig(dest)
    print 'Save', name, 'to', dest


def main():
    files = get_filenames()
    data = np.loadtxt(files.train, delimiter=SEP)
    test = np.loadtxt(files.test, delimiter=SEP)
    datasize = len(data)

    trainX, trainy = splitXY(data)
    testX, _ = splitXY(test)
    ytrue = test[:, -1:].ravel()


    x, y = [], []

    # generate the learning curve data
    for i in xrange(1, SAMPLE + 1):
        epochs = 6000 * i / SAMPLE
        # input size 64, hidden units 50, output size 10
        nn = NeuralNetwork([64, 50, 10], activator='sigmoid')
        nn.learn(trainX, trainy, epochs=epochs)
        yhat = [np.argmax(nn.classify(e)) for e in testX]

        check = [np.argmax(nn.classify(record[:-1])) == record[-1]
                 for record in test]
        counter = Counter(check)
        precision = counter[True] / float(counter[True] + counter[False])
        print 'epochs = %d,' % (epochs),
        print 'precision = %.4f' % (precision)
        x.append(epochs)
        y.append(precision)

    plot_curve(x, y, 'Epochs', 'Precision',
               'Learning curve', files.curve2)

    # x, y = [], []

    # # generate the learning curve data
    # for i in xrange(1, SAMPLE + 1):
    #     samplesize = datasize * i / SAMPLE
    #     sampleX, sampley = trainX[:samplesize], trainy[:samplesize]

    #     # input size 64, hidden units 50, output size 10
    #     nn = NeuralNetwork([64, 50, 10], activator='sigmoid')
    #     nn.learn(sampleX, sampley, epochs=10000)
    #     yhat = [np.argmax(nn.classify(e)) for e in testX]

    #     check = [np.argmax(nn.classify(record[:-1])) == record[-1]
    #              for record in test]
    #     counter = Counter(check)
    #     precision = counter[True] / float(counter[True] + counter[False])
    #     print 'training data size = %d,' % (samplesize),
    #     print 'precision = %.4f' % (precision)
    #     x.append(samplesize)
    #     y.append(precision)

    # plot_curve(x, y, 'Training data size', 'Precision',
    #            'Learning curve', files.curve)




if __name__ == "__main__":
    main()
