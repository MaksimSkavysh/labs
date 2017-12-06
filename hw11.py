import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.svm
import sklearn.neighbors


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


def counter_generator(init_value=0):
    counter = [init_value]

    def inner():
        counter[0] = counter[0] + 1
        return counter[0]
    return inner

counter = counter_generator()


def plot_distribution(x, num, name='distribution'):
    plt.figure(num)
    plt.hist(x)
    plt.title(name)


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


def plot_one_param(params, train_scores, dev_scores, title, param_name, figure_num):
    # plt.figure(counter(), figsize=(12, 12))
    plt.figure(figure_num)
    plt.plot(params, train_scores, label='train accuracy')
    plt.plot(params, dev_scores, label='dev accuracy')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('accuracy')
    plt.legend()
    # plt.show()


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


cool_digits = [0, 1, 2, 3, 4]
x_train, y_train = filter_by_digits(x_train, y_train, cool_digits, 5000)
plot_distribution(y_train, counter(), 'y_train distribution')

x_dev, y_dev = filter_by_digits(x_dev, y_dev, cool_digits, 1000)
plot_distribution(y_dev, counter(), 'y_dev distribution')

# x_train, y_train = X_train_dev, Y_train_dev
# x_dev, y_dev = X_test, Y_test


# Logistic regression
def train_logistic_regression(X_train, Y_train, X_dev, Y_dev):
    logistic_regression = sklearn.linear_model.LogisticRegression(
        multi_class='ovr',
        penalty='l2',  # the norm used in the penalization
        tol=0.01,  # default: 1e-4 Tolerance for stopping criteria.
        verbose=0,
        n_jobs=-1,
        solver='liblinear',  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    )
    logistic_regression.fit(X_train, Y_train)
    print('logistic_regression on train', logistic_regression.score(X_train, Y_train))
    print('logistic_regression on dev', logistic_regression.score(X_dev, Y_dev))


def train_naive_bayes(X_train, Y_train, X_dev, Y_dev):
    naive_bayes_estimators = [
        sklearn.naive_bayes.BernoulliNB,
        sklearn.naive_bayes.GaussianNB,
        sklearn.naive_bayes.MultinomialNB
    ]

    for estimator_class in naive_bayes_estimators:
        print("Using", estimator_class)
        estimator = estimator_class()
        estimator.fit(X_train, Y_train)
        print("Train score {:.2f}".format(estimator.score(x_train, y_train)))
        print("Dev score {:.2f}".format(estimator.score(X_dev, Y_dev)))


def train_model_with_params(model,
                            model_name,
                            params,
                            train_samples,
                            train_labels,
                            test_samples,
                            test_labels,
                            ):
    print('\n' + model_name + ':')
    train_scores = []
    dev_scores = []
    figure_num = counter()

    for param in params:
        estimator = model(param)
        estimator.fit(train_samples, train_labels)
        train_scores.append(estimator.score(train_samples, train_labels))
        dev_score = estimator.score(test_samples, test_labels)
        print('param: ', param, '; score: ', dev_score)
        dev_scores.append(dev_score)

    plot_one_param(
        params=params,
        train_scores=train_scores,
        dev_scores=dev_scores,
        title='SVC accuracy',
        param_name='kernel degree',
        figure_num=figure_num,
    )


# # Random Forest Classifier
# train_model_with_params(
#     model=lambda p: sklearn.ensemble.RandomForestClassifier(n_estimators=p, n_jobs=-1),
#     model_name='RandomForestClassifier',
#     params=range(50, 150, 20),
#     train_samples=x_train,
#     train_labels=y_train,
#     test_samples=x_dev,
#     test_labels=y_dev,
# )

# # SVC
# train_model_with_params(
#     model=lambda p: sklearn.svm.SVC(
#             kernel='poly',  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
#             degree=p,
#             coef0=0.0,  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
#             # shrinking=True,
#             # verbose=False,
#             # max_iter=-1,
#             decision_function_shape='ovr',  # 'ovo', 'ovr',
#         ),
#     model_name='SVC',
#     params=range(2, 4),
#     train_samples=x_train,
#     train_labels=y_train,
#     test_samples=x_dev,
#     test_labels=y_dev,
# )

# KNN
knn_model = lambda p: sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=p,
                weights='uniform',  # 'uniform' 'distance'
                leaf_size=30,
                n_jobs=-1,
            )
train_model_with_params(
    model=knn_model,
    model_name='KNN',
    params=range(2, 4),
    train_samples=x_train,
    train_labels=y_train,
    test_samples=x_dev,
    test_labels=y_dev,
)

# train_logistic_regression(x_train, y_train, x_dev, y_dev)

# train_naive_bayes(x_train, y_train, x_dev, y_dev)

# help(sklearn.neighbors.KNeighborsClassifier)

plt.show()


