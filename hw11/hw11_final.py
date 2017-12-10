import utils
from algorithms import evaluate_models_training
from algorithms import train_logistic_regression
from algorithms import train_naive_bayes
import matplotlib.pyplot as plt
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.svm

MNIST_WATERLINE = 60000

DIGITS = [0, 1, 2, 3, 4, 5]
FILTERED_TRAIN_DATA = 3000
FILTERED_TEST_DATA = 2000

grapher = utils.Grapher()
mnist_data = utils.MinstDataManager()
mnist_data.prepare_mnist_data('./mnist')

# x_train, y_train = utils.filter_by_digits(mnist_data.x_train, mnist_data.y_train, DIGITS, FILTERED_TRAIN_DATA)
# x_test, y_test = utils.filter_by_digits(mnist_data.x_test, mnist_data.y_test, DIGITS, FILTERED_TEST_DATA)


def plot_distributions():
    grapher.plot_distribution(y_train, name='y_train distribution')
    grapher.plot_distribution(y_test, name='y_test distribution')
    grapher.show()


print('Starting...\n')
model = sklearn.svm.SVC(
    kernel='poly',  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree=2,
    coef0=0.0,  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    tol=0.003,
    # gamma=0.0001,
    # cache_size=500,
    # shrinking=True,
    verbose=True,
    # max_iter=-1,
    decision_function_shape='ovr',  # 'ovo', 'ovr',
)


x_train, y_train = mnist_data.x_train, mnist_data.y_train

grapher.visualize(x_train[0], title='before deskewing')
x_train = utils.deskewAll(x_train)
grapher.visualize(x_train[0], title='after deskewing')

model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
print('Accuracy on the train data:' + str(train_score))


x_test, y_test = mnist_data.x_test_mnist, mnist_data.y_test_mnist
x_test = utils.deskewAll(x_test)
test_score = model.score(x_test, y_test)
print('Accuracy on the mnist original test data:' + str(test_score))

grapher.show()
