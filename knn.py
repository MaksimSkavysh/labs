import math
import numpy as np


def getClass(name):
    if name == 'Iris-setosa':
        return 1.0
    elif name == 'Iris-versicolor':
        return 2.0
    elif name == 'Iris-virginica':
        return 3.0
    print(name)
    raise NameError('incorrect iris')


def get_data(address, shuffle=True):
    s = []
    with open(address) as file:
        for line in file:
            parts = line.split(",")
            if len(parts) > 4:
                last = parts[-1].strip()
                parts[-1] = getClass(last)
                s.append([float(x) for x in parts])
    if shuffle:
        np.random.shuffle(s)
    return s


def split_data(samples, rate=0.9):
    train_num = int(len(samples)*rate)
    return samples[:train_num], samples[train_num:]


def get_class_counters(samples):
    counters = {}
    for s in samples:
        if s[-1] in counters:
            counters[s[-1]] += 1
        else:
            counters[s[-1]] = 1
    return counters


def euclidean_distance(x1, x2):
    distance = 0
    length = len(x1)
    for i in range(0, length, 1):
        try:
            distance += pow((x1[i] - x2[i]), 2)
        except TypeError:
            print(x1[i])
    return math.sqrt(distance)


def get_neighbors(samples, test, k):
    distances = []

    for s in samples:
        dist = euclidean_distance(test, s)
        distances.append((s[-1], dist))
    distances.sort(key=lambda el: el[-1])
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors


def cross_validation_split(dataset, n_folds):
    folds_data = list()
    fold_length = math.ceil(len(dataset)/n_folds)
    for i in range(0, n_folds):
        folds_data.append(dataset[fold_length*i: fold_length*(i + 1)])
    return folds_data


def run_cross_validation(samples, k, n_folds=10):
    folds = cross_validation_split(samples, n_folds)

    accuracy = 0

    for i in range(0, len(folds)):
        test_fold = folds[i]
        train_x = []
        for j in range(0, len(folds)):
            if i != j:
                train_x = train_x + folds[j]
        accuracy = accuracy + 0
    accuracy = accuracy/n_folds
    return accuracy, 0


def main():
    samples = get_data("./data/iris.data.txt")
    train, test = split_data(samples)

    print('\nCounters:')
    print(get_class_counters(train))
    print(get_class_counters(test))
    print('-------------------------------------------------------------------------------------------\n')

    print(get_neighbors(train, test[0], 2))
    print(test[0])

    run_cross_validation(samples, 1)

main()
