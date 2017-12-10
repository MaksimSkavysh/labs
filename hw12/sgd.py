import random
import math
from math import exp
import numpy as np



def get_data(address, currentIris):
    samples = []
    with open(address) as file:
        for line in file:
            parts = line.split(",")
            last = parts[-1]
            if last.strip() == currentIris:
                parts[-1] = 1
            else:
                parts[-1] = 0
            samples.append([float(x) for x in parts])
    return samples


def split_data(samples, percent=0.9):
    indexes = [i for i in range(0, len(samples), 1)]
    random.shuffle(indexes)
    train_part = int(len(samples)*percent)
    trainings = []
    tests = []
    for i in indexes[0:train_part]:
        trainings.append(samples[i])
    for i in indexes[train_part:]:
        tests.append(samples[i])
    return trainings, tests


def true_log(x):
    return math.log(x)


def sigmoid(z):
    return float(1.0 / float((1.0 + exp(-1.0*z))))


def predict(row, w):
    result = w[0]
    for i in range(len(row)-1):
        result += w[i + 1] * row[i]
    # if result > 0:
    #     return 1.0
    # else:
    #     return -1.0
    return result


def zero_one_loss(result):
    if result > 0:
        return 1.0
    else:
        return 0.0


def loss(sample, w):
    pred = predict(sample, w)
    # print(sigmoid(pred))
    return sigmoid(pred) - sample[-1]


def gradient(sample, error, m, eta):
    v_t = [1, *sample[0:-1]]
    v_t = np.multiply(v_t, - (1 / m) * eta * error)
    return v_t


def coefficients_sgd(samples, eta, T):
    w = [0.0 for i in range(len(samples[0]))]
    m = len(samples)
    for t in range(0, T, 1):
        for sample in samples:
            error = loss(sample, w)
            v_t = gradient(sample, error, m, eta)
            w = np.add(w, v_t)
    return w



currentIris = 'Iris-setosa'
# currentIris = 'Iris-versicolor'
# currentIris = 'Iris-virginica'

irises = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


for currentIris in irises:
    print(currentIris)
    samples = get_data("./data/iris.data.txt", currentIris)
    trainings, tests = split_data(samples)

    eta = 0.5
    T = 10
    # tau = 0.5
    w = coefficients_sgd(samples, eta, T)
    sum_err = 0
    for test in tests:
        predicted = zero_one_loss(predict(test, w))
        sum_err = sum_err + abs(test[-1] - predicted)

    print('loss: ', sum_err/len(tests))


