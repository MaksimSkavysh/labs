import random
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


def plot_distribution(x, num, name='distribution'):
    plt.figure(num)
    plt.hist(x)
    plt.title(name)


def counter_generator(init_value=0):
    counter = [init_value]

    def inner():
        counter[0] = counter[0] + 1
        return counter[0]
    return inner


def get_class_counters(samples):
    counters = {}
    for s in samples:
        if s[-1] in counters:
            counters[s[-1]] += 1
        else:
            counters[s[-1]] = 1
    return counters


def visualize(x, title=None):
    x = np.array(x, dtype='uint8').reshape((28, 28))
    if title:
        plt.title(title)
    plt.imshow(x, cmap='gray')
    plt.show()


def filter_by_digits(x, y, digits, limit=None):
    indices = [i for i in range(len(x)) if y[i] in digits]
    limit = limit or len(indices)
    return x[indices][:limit], y[indices][:limit]


def get_data(home):
    mnist = fetch_mldata('MNIST original', data_home=home)
    X, Y = mnist.data, np.array(mnist.target, dtype='int')
    return X, Y


counter = counter_generator()
ntrain_dev = 60000

X, Y = get_data('./mnist')
X_train_dev, Y_train_dev = X[:ntrain_dev], Y[:ntrain_dev]
X_test, Y_test = X[ntrain_dev:], Y[ntrain_dev:]


ntrain = int(ntrain_dev * 90 / 100)
permutation = np.random.permutation(ntrain_dev)
x_train = X_train_dev[permutation][:ntrain]
y_train = Y_train_dev[permutation][:ntrain]
x_dev = X_train_dev[permutation][ntrain:]
y_dev = Y_train_dev[permutation][ntrain:]


cool_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x_train, y_train = filter_by_digits(x_train, y_train, cool_digits, 10000)
plot_distribution(y_train, counter(), 'y_train distribution')

x_dev, y_dev = filter_by_digits(x_dev, y_dev, cool_digits, 1000)
plot_distribution(y_dev, counter(), 'y_dev distribution')

# x_test, y_test = filter_by_digits(X_test, Y_test, cool_digits, 100)
# plot_distribution(y_test, counter(), 'y_test distribution')


# plt.figure(counter())
# num = random.randint(0, len(x_train) - 1)
# visualize(x_train[num], title='This is {}'.format(y_train[num]))


# Logistic regression
# logistic_regression = sklearn.linear_model.LogisticRegression()
# logistic_regression.fit(x_train, y_train)
# print('logistic_regression on train', logistic_regression.score(x_train, y_train))
# print('logistic_regression on dev', logistic_regression.score(x_dev, y_dev))


# Random forest
def plot_one_param(params, train_scores, dev_scores, title, param_name):
    # plt.figure(counter(), figsize=(12, 12))
    plt.figure(counter())
    plt.plot(params, train_scores, label='train accuracy')
    plt.plot(params, dev_scores, label='dev accuracy')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def train_random_forest(n_estimators, X_train, Y_train, X_dev, Y_dev):
    train_scores = []
    dev_scores = []

    for estimators in n_estimators:
        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=estimators, n_jobs=-1)
        clf.fit(X_train, Y_train)
        train_scores.append(clf.score(X_train, Y_train))
        dev_scores.append(clf.score(X_dev, Y_dev))

    plot_one_param(
        params=n_estimators,
        train_scores=train_scores,
        dev_scores=dev_scores,
        title='Random forest accuracy',
        param_name='number of trees'
    )

train_random_forest(range(10, 300, 10), x_train, y_train, x_dev, y_dev)


plt.show()