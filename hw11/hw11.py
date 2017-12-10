import utils
from algorithms import evaluate_models_training
from algorithms import train_logistic_regression
from algorithms import train_naive_bayes
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.svm
import sklearn.neighbors

MNIST_WATERLINE = 60000

DIGITS = [0, 1, 2, 3, 4, 5]
FILTERED_TRAIN_DATA = 5000
FILTERED_TEST_DATA = 2000

grapher = utils.Grapher()
mnist_data = utils.MinstDataManager()
mnist_data.prepare_mnist_data('./mnist')

x_train, y_train = utils.filter_by_digits(mnist_data.x_train, mnist_data.y_train, DIGITS, FILTERED_TRAIN_DATA)
x_test, y_test = utils.filter_by_digits(mnist_data.x_test, mnist_data.y_test, DIGITS, FILTERED_TEST_DATA)


def plot_distributions():
    grapher.plot_distribution(y_train, name='y_train distribution')
    grapher.plot_distribution(y_test, name='y_test distribution')
    grapher.show()


def run_test_and_print(model_geter, params_list, title='model', param_name='params'):
    train_scores, test_scores = evaluate_models_training(
        model_geter=model_geter,
        params_list=params_list,
        train_samples=x_train,
        train_labels=y_train,
        test_samples=x_test,
        title=title,
        test_labels=y_test,
    )
    grapher.plot_one_param(
        params_list=params_list,
        train_scores=train_scores,
        test_scores=test_scores,
        title=title,
        param_name=param_name,
    )


# # Logistic regression
# logistic_regression = train_logistic_regression(x_train, y_train)
# print('logistic_regression on test data set', logistic_regression.score(x_test, y_test))


# # Naive bayes
# naive_bayes_estimators = train_naive_bayes(x_train, y_train, x_test, y_test)
#
#

def RandomForests():
    title = 'Random forest'
    params_list = range(1, 100, 10)
    param_name = 'Number of trees estimators'
    model_geter = lambda estimators: sklearn.ensemble.RandomForestClassifier(
            n_estimators=estimators,
            criterion='gini',  # Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain
            max_depth=None,
            min_samples_split=2,  # The minimum number of samples required to split an internal node
            n_jobs=-1
        )
    run_test_and_print(model_geter, params_list, title, param_name)


def SVC():
    title = 'SVC'
    params_list = range(1, 5, 1)
    param_name = 'SVC polinom degree'
    model_geter = lambda degree: sklearn.svm.SVC(
                kernel='poly',  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
                degree=degree,
                coef0=0.0,  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
                tol=0.01,
                # cache_size=500,
                # shrinking=True,
                # verbose=False,
                # max_iter=-1,
                decision_function_shape='ovr',  # 'ovo', 'ovr',
            )
    run_test_and_print(model_geter, params_list, title, param_name)


def KNN():
    title = 'KNN'
    params_list = range(1, 5, 1)
    param_name = 'KNN neighbors'
    model_geter = lambda n_neighbors: sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights='uniform',  # 'uniform' 'distance'
                leaf_size=30,
                n_jobs=-1,
            )
    run_test_and_print(model_geter, params_list, title, param_name)


# plot_distributions()

SVC()
KNN()
RandomForests()

grapher.show()
