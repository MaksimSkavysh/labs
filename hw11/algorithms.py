import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.svm
import sklearn.neighbors


def evaluate_models_training(model_geter, params_list, train_samples,
                             train_labels, test_samples, test_labels, title='model', verbose=True):
    train_scores = []
    test_scores = []
    if verbose:
        print('\nevaluete: ' + title)
    for param in params_list:
        if verbose:
            print('param: ', str(param))
        clf = model_geter(param)
        clf.fit(train_samples, train_labels)
        train_score = clf.score(train_samples, train_labels)
        train_scores.append(train_score)
        test_score = clf.score(test_samples, test_labels)
        test_scores.append(test_score)
        if verbose:
            print('test:' + str(test_score))
            print('train:' + str(train_score))

    return train_scores, test_scores


def train_logistic_regression(train_samples, train_labels):
    logistic_regression = sklearn.linear_model.LogisticRegression(
        multi_class='ovr',
        penalty='l2',  # the norm used in the penalization
        tol=0.01,  # default: 1e-4 Tolerance for stopping criteria.
        verbose=0,
        n_jobs=-1,
        solver='saga',  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    )
    logistic_regression.fit(train_samples, train_labels)
    print('logistic_regression on train data set', logistic_regression.score(train_samples, train_labels))
    return logistic_regression


def train_random_forest(train_samples, train_labels, n_estimators):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    clf.fit(train_samples, train_labels)
    print('random forest on train data set', clf.score(train_samples, train_labels))
    return clf


def train_naive_bayes(train_samples, train_labels, test_samples, test_labels):
    naive_bayes_estimators = [
        sklearn.naive_bayes.BernoulliNB,
        sklearn.naive_bayes.GaussianNB,
        sklearn.naive_bayes.MultinomialNB
    ]

    for estimator_class in naive_bayes_estimators:
        print("Using", estimator_class)
        estimator = estimator_class()
        estimator.fit(train_samples, train_labels)
        print("Train score {:.2f}".format(estimator.score(train_samples, train_labels)))
        print("Test score {:.2f}".format(estimator.score(test_samples, test_labels)))
    return naive_bayes_estimators
