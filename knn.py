from math import exp
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


samples = get_data("./data/iris.data.txt")
print(samples[0:10])
