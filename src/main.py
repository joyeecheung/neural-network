#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

from util import get_filenames
from network import NeuralNetwork

SEP = ','
SAMPLE = 30
TOTAL_EPOCH = 50000


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
    print 'Mean of %s = %.4f' % (ylabel, ymean)
    print 'Standard deviation of %s = %.4f' % (ylabel, ystd)
    print 'Min = %.4f, max = %.4f' % (ymin, ymax)

    xy = sorted(zip(x, y), key=lambda a: a[0])
    x, y = zip(*xy)

    plt.figure()
    # setup decorations
    plt.rc('font', family='serif')

    # plt.yticks(np.arange(ylim[0], ylim[1], ylim[1]))
    # plt.ylim(0.0, 1.0)
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
    txt += '\n$min = %.4f$, $max = %.4f$' % (ymin, ymax)
    pltymin, pltymax = plt.ylim()
    pltxmin, pltxmax = plt.xlim()
    txtx = pltxmin + (pltxmax - pltxmin) * 0.5
    txty = pltymin + (pltymax - pltymin) * 0.5
    plt.text(txtx, txty, txt, bbox=box)

    plt.savefig(dest)
    print 'Save', name, 'to', dest


def main():
    files = get_filenames()
    data = np.loadtxt(files.train, delimiter=SEP)
    test = np.loadtxt(files.test, delimiter=SEP)
    datasize = len(data)

    trainX, trainy = splitXY(data)
    testX, _ = splitXY(test)

    x, y = [], []
    epochs = 0
    nn_epochs = NeuralNetwork([64, 50, 10], activator='logistic')

    # generate the learning curve data
    while epochs < TOTAL_EPOCH:
        epochs += TOTAL_EPOCH / SAMPLE
        # input size 64, hidden units 50, output size 10
        nn_epochs.learn(trainX, trainy, epochs=TOTAL_EPOCH / SAMPLE)
        yhat = [np.argmax(nn_epochs.classify(e)) for e in testX]

        check = [np.argmax(nn_epochs.classify(record[:-1])) == record[-1]
                 for record in test]
        counter = Counter(check)
        print 'epochs = %d,' % (epochs),
        print 'Total errors = %.4f' % (counter[False])
        x.append(epochs)
        y.append(counter[False])

    plot_curve(x, y, 'Epochs', 'Total errors',
               'Errors decreased with epochs', files.error)

    x, y = [], []

    # generate the learning curve data
    for i in xrange(1, SAMPLE + 1):
        samplesize = datasize * i / SAMPLE
        sampleX, sampley = trainX[:samplesize], trainy[:samplesize]

        # input size 64, hidden units 50, output size 10
        nn = NeuralNetwork([64, 50, 10], activator='logistic')
        nn.learn(sampleX, sampley, epochs=TOTAL_EPOCH)
        yhat = [np.argmax(nn.classify(e)) for e in testX]

        check = [np.argmax(nn.classify(record[:-1])) == record[-1]
                 for record in test]
        counter = Counter(check)
        precision = counter[True] / float(counter[True] + counter[False])
        print 'training data size = %d,' % (samplesize),
        print 'precision = %.4f' % (precision)
        x.append(samplesize)
        y.append(precision)

    plot_curve(x, y, 'Training data size', 'Precision',
               'Learning curve', files.curve)


if __name__ == "__main__":
    main()
