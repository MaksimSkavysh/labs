import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from scipy.ndimage import interpolation


MNIST_WATERLINE = 60000


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


def get_data(home='./mnist'):
    mnist = fetch_mldata('MNIST original', data_home=home)
    x, y = mnist.data, np.array(mnist.target, dtype='int')
    return x, y


def split_data(samples, labels, waterline=60000):
    train_samples, train_labels = samples[:waterline], labels[:waterline]
    test_samples, test_labels = samples[waterline:], labels[waterline:]
    return train_samples, train_labels, test_samples, test_labels


def filter_by_digits(x, y, digits, limit=None):
    indices = [i for i in range(len(x)) if y[i] in digits]
    limit = limit or len(indices)
    return x[indices][:limit], y[indices][:limit]


def moments(image):
    c0, c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28, 28)).flatten())
    return np.array(currents)


class MinstDataManager:
    def __init__(self):
        # Original data
        # self.x_mnist = []
        # self.y_mnist = []

        # Original data splited on test/train
        self.x_tr_mnist = []
        self.y_tr_mnist = []
        self.x_test_mnist = []
        self.y_test_mnist = []

        # Original train data splited on test/train
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def prepare_mnist_data(self, path='./mnist', n_train=int(MNIST_WATERLINE * 90 / 100)):
        x_mnist, y_mnist = get_data(path)
        # self.x_mnist = x_mnist
        # self.y_mnist = y_mnist
        self.x_tr_mnist, \
            self.y_tr_mnist, \
            self.x_test_mnist, \
            self.y_test_mnist = split_data(x_mnist, y_mnist, MNIST_WATERLINE)

        permutation = np.random.permutation(MNIST_WATERLINE)

        self.x_train, \
            self.y_train, \
            self.x_test, \
            self.y_test = split_data(self.x_tr_mnist[permutation], self.y_tr_mnist[permutation], n_train)



class Grapher:
    def __init__(self):
        self.counter = counter_generator()

    def plot_distribution(self, x, name='distribution', new_figure=True):
        if new_figure:
            plt.figure(self.counter())
        plt.hist(x)
        plt.title(name)

    def visualize(self, x, title=None, new_figure=True):
        if new_figure:
            plt.figure(self.counter())
        x = np.array(x, dtype='uint8').reshape((28, 28))
        if title:
            plt.title(title)
        plt.imshow(x, cmap='gray')

    def plot_one_param(self,
                       params_list,
                       train_scores,
                       test_scores,
                       title=None,
                       param_name='parameters',
                       new_figure=True):
        if new_figure:
            plt.figure(self.counter())
        plt.plot(params_list, train_scores, label='train accuracy')
        plt.plot(params_list, test_scores, label='dev accuracy')
        plt.title(title)
        plt.xlabel(param_name)
        plt.ylabel('accuracy')
        plt.legend()
        # plt.show()

    @staticmethod
    def show():
        plt.show()



