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
                s.append(parts)
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
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def get_neighbors(samples, test, k):
    distances = []

    for s in samples:
        dist = euclidean_distance(test, s)
        distances.append((s, dist))
    print(distances)

    # distances.sort()
    neighbors = []
    # for x in range(k):
    #     neighbors.append(distances[x][0])
    return neighbors


def main():
    samples = get_data("./data/iris.data.txt")
    train, test = split_data(samples)

    print('\nCounters:')
    print(get_class_counters(train))
    print(get_class_counters(test))
    print('--------------------------------------------------------------------------------------------')

    get_neighbors(samples[0:5], test[0], 2)

main()
