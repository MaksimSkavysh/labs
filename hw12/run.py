import random
import math
from math import exp
import numpy as np

# currentIris = 'Iris-setosa'
currentIris = 'Iris-versicolor'


# currentIris = 'Iris-virginica'


def get_data(address):
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
    train_part = int(len(samples) * percent)
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
    return float(1.0 / float((1.0 + exp(-1.0 * z))))


def predict(row, w):
    result = w[0]
    for i in range(len(row) - 1):
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


def linear_regression_sgd(train, test, l_rate, eta, T):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return (predictions)


def cross_validation_split(samples, k_folds):
    samples_split = list()
    samples_copy = list(samples)
    fold_size = int(len(samples) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(samples_copy))
            fold.append(samples_copy.pop(index))
        samples_split.append(fold)
    return samples_split


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return math.sqrt(mean_error)


def evaluate_algorithm(samples, k_folds, eta, T):
    folds = cross_validation_split(samples, k_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = coefficients_sgd(train_set, test_set, eta, T)
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    return scores


samples = get_data("./data/iris.data.txt")

eta = 0.5
T = 10
scores = evaluate_algorithm(samples, 5, eta, T)

print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores) / float(len(scores))))

# tau = 0.5
# trainings, tests = split_data(samples)

# w = coefficients_sgd(samples, eta, T)
# print(w)
# sum_err = 0
# for test in tests:
#     predicted = zero_one_loss(predict(test, w))
#     sum_err = sum_err + abs(test[-1] - predicted)
#     print(predicted, test[-1])
#
# print(sum_err/len(tests))


