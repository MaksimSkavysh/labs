import numpy as np
from scipy.special import expit

sample_size, features_count = 25, 10
w = np.random.random(features_count)
X, y = np.random.random((sample_size, features_count)), 2 * (np.random.randint(0, 2, sample_size) - 0.5)

XY = np.hstack((X, y[:, np.newaxis]))
del X, y

X, y = XY[:, np.arange(0, features_count)], XY[:, features_count]


def logistic(w, X, y):
    N = X.shape[0]
    C = 0.001
    reg = C/2 * (np.linalg.norm(w) ** 2)
    log_array = np.logaddexp(np.zeros(N), np.multiply(-y, np.matmul(X, w)))
    log_regression_quality = 1/N * np.sum(log_array) + reg
    return log_regression_quality


def logistic_grad(w, X, y):
    N = X.shape[0]
    M = w.shape[0]
    grad = np.zeros(M)
    h = expit(np.matmul(X, w))
    # h = logistic(w, X, y)
    delta = h - y
    # grad = np.multiply(-(1.0 / N), np.matmul(X.T, delta))
    grad = np.multiply(-(1.0 / N), np.matmul(X.T, delta))
    return grad


def max_error(a, b):
    return np.max(np.abs(a - b))


def grad_finite_diff(func, x, eps=1e-8):
    N = x.shape[0]
    E = np.identity(N)
    x, fval, dnum = x.astype(np.float64), func(x), np.zeros_like(x)
    dnum = np.apply_along_axis(lambda e: (func(x + np.multiply(eps, e)) - fval) / eps, 0, E)
    return dnum

# logistic(w, X, y)
# logistic_grad(w, X, y)

mat_grad = logistic_grad(w, X, y)
num_grad = grad_finite_diff(lambda x: logistic(x, X, y), w)

for i in range(0, 5):
    print(mat_grad[i]/num_grad[i])
    # num_grad[i] = num_grad[i] * -2.27746893963

err = max_error(mat_grad, num_grad)
print('err = ', err, 'ok' if err < 1e-6 else 'ошибка очень большая!')


def logistic_hess(w, X, y):
    """
        logistic_hess(w, X, y) вычисляет гессиан функции качества лог регрессии dL(w, X, y)/dw

        w: np.array размера (M,)
        X: np.array размера (N, M)
        y: np.array размера (M,)

        hessw: np.array размера (M, M)
    """
    # hessw = # Гессиан dL/dw_iw_j
    # YOUR CODE HERE
    raise NotImplementedError()

assert(logistic_hess(w, X, y).shape == (w.shape[0], w.shape[0]))

print('\n')
