import random
from math import exp
import numpy as np

currentIris = 'Iris-setosa'


def get_data(address):
    samples = []
    labels = []
    with open(address) as file:
        for line in file:
            parts = line.split(",")
            last = parts[-1]
            samples.append([float(x) for x in parts[0:-1]])
            if last.strip() == currentIris:
                labels.append(1)
            else:
                labels.append(0)
    return samples, labels


def split_data(samples, labels, percent=0.9):
    indexes = [i for i in range(0, len(samples), 1)]
    random.shuffle(indexes)
    train_part = int(len(samples)*percent)
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    for i in indexes[0:train_part]:
        train_samples.append(samples[i])
        train_labels.append(labels[i])
    for i in indexes[train_part:]:
        test_samples.append(samples[i])
        test_labels.append(labels[i])
    return train_samples, train_labels, test_samples, test_labels, indexes


def predict(row, coefficients):
    label = coefficients[0]
    for i in range(len(row)-1):
        label += coefficients[i + 1] * row[i]
    return label


def sigma(x):
    return 1/(1+exp(-x))


def coefficients_sgd(samples, eta, T):
    w = [0.0]*4
    m = len(samples)
    for t in range(0, T, 1):
        for sample in samples:
            error = predict(sample, w) - sample[-1]
            w[0] = w[0] - (1/m) * eta * error
            for i in range(len(sample)-1):
                w[i + 1] = w[i + 1] - (1/m) * eta * error * sample[i]
    return w



samples, labels = get_data("./data/iris.data.txt")
tr_s, tr_l, test_s, test_l, indexes = split_data(samples, labels)


# eta = 0.5
# T = 10
# tau = 0.5


print(tr_s[0])
print(test_l)
print(len(labels), sum(labels))


