import utils
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.svm
import sklearn.neighbors

MNIST_WATERLINE = 60000
DIGITS = [0, 1, 2, 3, 4]


grapher = utils.Grapher()

x_mnist, y_mnist = utils.get_data('./mnist')
x_tr_mnist, y_tr_mnist, x_test_mnist, y_test_mnist = utils.split_data(x_mnist, y_mnist, MNIST_WATERLINE)

ntrain = int(MNIST_WATERLINE * 90 / 100)
permutation = np.random.permutation(MNIST_WATERLINE)
x_train, y_train, x_test, y_test = utils.split_data(x_tr_mnist[permutation], y_tr_mnist[permutation], ntrain)

x_train, y_train = utils.filter_by_digits(x_train, y_train, DIGITS, 10000)
grapher.plot_distribution(y_train, name='y_train distribution')

x_test, y_test = utils.filter_by_digits(x_test, y_test, DIGITS, 1000)
grapher.plot_distribution(y_test, name='y_test distribution')


def train_logistic_regression(train_samples, train_labels):
    logistic_regression = sklearn.linear_model.LogisticRegression(
        multi_class='ovr',
        penalty='l2',  # the norm used in the penalization
        tol=0.01,  # default: 1e-4 Tolerance for stopping criteria.
        verbose=0,
        n_jobs=-1,
        solver='liblinear',  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    )
    logistic_regression.fit(train_samples, train_labels)
    print('logistic_regression on train data set', logistic_regression.score(train_samples, train_labels))
    return logistic_regression


logistic_regression = train_logistic_regression(x_test, y_test)
print('logistic_regression on train data set', logistic_regression.score(x_test, y_test))